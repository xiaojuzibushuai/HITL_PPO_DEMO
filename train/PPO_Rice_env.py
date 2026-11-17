"""水稻田逐日灌溉仿真环境（一期正式版，训练 / 评估共用）。

一期正式环境模块；请 ``from train.PPO_Rice_env import Irrigation``。
相对路径 forecast/actual/growth_para 由 data_prep 拷入 cwd。
日历与 RAW/TAW 只从 ``configs/scenario.json`` 读（``_sc()``），禁止写死。
"""
from __future__ import annotations

import datetime
import sys
from math import prod
from pathlib import Path

import numpy as np
import pandas as pd

# 情景参数（土壤/日历）统一从 configs/scenario.json 读
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from common.scenario import get_scenario, ks_factor  # noqa: E402

np.random.seed(2)

# Bump when the reward semantics change; record alongside every trained checkpoint.
REWARD_PROTOCOL_VERSION = "2026.10.26-reward-v1"


def _sc():
    """每次取当前情景（支持 --scenario 清缓存后生效，禁止 import 时写死）。"""
    return get_scenario()


def _md_to_ymd(year: str | int, md: str) -> str:
    """scenario 的 MM/DD → 与 csv 一致的无前导零 Y/M/D，如 2100/6/15。"""
    m, d = md.split("/")
    return f"{int(year)}/{int(m)}/{int(d)}"


def _year_of_date(date_val) -> str:
    return str(date_val).split("/")[0]


class Irrigation:
    """单田块逐日环境。读的是「当前工作目录」下的相对文件名。

    训练 / 评估表可以不同，但入口已用拷贝统一成同名文件：
      - 训练：`prepare_train_workdir` 按 scenario.forecast/actual 拷表（开训前全年份校验）
      - 评估：`prepare_eval_workdir` 按 scenario.forecast_eval/actual_eval 拷表

    故本类始终 `read_csv('forecast.csv')`；换数据集靠换 workdir 内容，不靠换文件名。
    `state_space` 与 `state_space_policy` 各读一份独立 DataFrame，避免互相改脏（历史字段名保留）。
    """

    # 仅缓存 growth_para（相对 cwd 读）；forecast/actual 每次 __init__ 新读，训/评表不会混
    _growth_para_cache = None

    def __init__(self):
        Irrigation._growth_para_cache = None  # 新实例按当前 cwd 重载生育期表
        self.state_space = pd.read_csv("forecast.csv", header=0)
        self.actual_state_space = pd.read_csv("actual.csv", header=0)
        self.state_space_policy = pd.read_csv("forecast.csv", header=0)
        self.actual_state_space_policy = pd.read_csv("actual.csv", header=0)

    @classmethod
    def _load_growth_para(cls):
        if cls._growth_para_cache is None:
            gp = pd.read_csv("growth_para.csv", header=0)
            cls._growth_para_cache = gp.set_index(["period_name"])
        return cls._growth_para_cache

    @staticmethod
    def get_growth_stage(date):
        """获取当前生育阶段（生育期表缓存，勿每步读盘）。"""
        gp = Irrigation._load_growth_para()
        growth_date = datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%m/%d')
        transplanting_day = datetime.datetime.strptime(gp.loc['start', 'Transplanting'], '%m/%d')
        growth_date_obj = datetime.datetime.strptime(growth_date, '%m/%d')
        delta = growth_date_obj - transplanting_day
        transplanting_day_ = delta.days + 1  # 加 1 是因为从第 1 天开始计算
        for each in gp.columns:  # 各生育期
            date_end = datetime.datetime.strptime(gp.loc['end', each], '%m/%d')
            if growth_date_obj <= date_end:
                return each, transplanting_day_
        return "Unknown", transplanting_day_

    @staticmethod
    def calculate_days_difference(state, start):
        """本季第几天（0-based）：state.date 相对 scenario.start（MM/DD），忽略年份。"""
        date = datetime.datetime.strptime(str(state.date), "%Y/%m/%d").replace(year=1900)
        start_date = datetime.datetime.strptime(start, "%m/%d").replace(year=1900)
        return (date - start_date).days

    @staticmethod
    def process_series(state):
        """Series → 网络输入 list（去掉 date / h_after / p，保留 8 维状态列顺序）。"""
        data = state.drop(["date", "h_after", "p"])
        return list(data)


    def reward(self, actual_state_pd, state_space, state, actual_state, action, i):
        """逐步奖励（环境内）。返回 (r_env, r0, r1, r2, r4)。

        合成：r_env = r0 * r1 * r2 * r3 * r4 * r5，其中 r1 再 min(., 1)。
        训练外层（train_ppo）还会：减非零动作罚、季末动作次数罚、再 ×0.1。

        智能体如何学到东西（直觉）：
          - 同一状态下，合规动作 → r0≈10，乱动作 → r0≈1，乘积差约 10 倍；
          - 过干 → r2 掉；超淹 → r4 掉；浪费水 → r1 掉；大动作 → r5 掉；
          - PPO 提高「高回报轨迹」里动作的概率，从而学会「该灌灌、该排排、中间少动」。

        参数 i：本季第几天（从 start 起算的天数差），用于 7 日前瞻；当年表用 iloc，勿用全表 loc。
        """
        sc = _sc()
        RAW, TAW, Duration, DEFAULT_PERC = sc.raw, sc.taw, sc.duration_days, sc.default_perc_mm_d
        # 当年 forecast / actual：按季内日序；全表 loc[i] 会串年
        year = str(state.date).split("/")[0]
        fc = state_space[
            state_space["date"].astype(str).str.split("/").str[0] == year
        ].reset_index(drop=True)
        act = actual_state_pd.reset_index(drop=True)

        def _fc_at(day_idx: int):
            if day_idx < 0 or day_idx >= len(fc):
                return fc.iloc[-1]
            return fc.iloc[day_idx]

        def _act_at(day_idx: int):
            if day_idx < 0 or day_idx >= len(act):
                return act.iloc[-1]
            return act.iloc[day_idx]

        def cal_md(action):
            """把动作拆成灌水量 m≥0、排水量 d≤0（互斥）。"""
            m, d = 0, 0
            if action > 0:
                m = action
            elif action < 0:
                d = action
            return m, d

        def cal_r0(i, actual_state_pd, state_space, state, actual_state, m, d):
            """r0：农艺合规大档（约 10 合规 / 1 不合规）。

            | 水位区间              | 偏好           | 合规 r0 | 乱做 r0 |
            |----------------------|----------------|---------|---------|
            | h < hmin             | 宜灌或不动     | 10      | 排→1    |
            | h ≥ Hc               | 宜排           | 10      | 灌/不动→1 |
            | hmin ≤ h ≤ Hp        | 宜不动         | 10      | 灌或排→1 |
            | Hp < h < Hc          | 宜排或不动     | 10      | 灌→1    |
            """
            r0 = 0
            if state.h_before < state.hmin:
                if d < 0:
                    r0 = 1
                elif m >= 0:
                    r0 = 10
            elif state.h_before >= state.Hc:
                # 超 Hc 时，排水得合规奖励；不动或灌水均为不合规。
                # m>=0 恒成立，不能用它判断动作方向，否则排水分支永远不可达。
                if d < 0:
                    r0 = 10
                else:
                    r0 = 1
            elif state.hmin <= state.h_before <= state.Hp:
                if m > 0 or d < 0:
                    r0 = 1
                elif m == 0 and d == 0:
                    r0 = 10
            elif state.h_before > state.Hp:
                if m > 0:
                    r0 = 1
                elif d <= 0:
                    r0 = 10
            return r0

        def cal_r1r2(i, actual_state_pd, state_space, state, actual_state, m, d):
            """灌侧 / 适灌带内：7 日前瞻 → r1 用水效率、r2 产量水分因子；r3/r4 置 1。

            r1 ≈ (有效蒸腾 ETC + 渗漏) / (雨 + 灌 − 期末水位变化相关项)，鼓励「雨尽其用」。
            r2 ≈ Π (ks)^λ，过干 ks↓ → 产量因子掉。
            """
            range_num = 7 if (i  <= (Duration - 7)) else 7 - (i - (Duration - 7))
            range_num = max(1, min(range_num, len(fc) - i, len(act) - i))

            r3, r4 = 1, 1
            da, ks, h, h_, lam = [0.0] * range_num, [0.0] * range_num, [0.0] * range_num, [0.0] * range_num, [0.0] * range_num
            ETC, ET, pp, P = 0.0, 0.0, 0, 0

            if state.h_before < -TAW:
                return 0, 0, 0, 0
            for each in range(range_num):
                row_f = _fc_at(i + each)
                row_a = _act_at(i + each)
                if each == 0:
                    da[each] = d
                    h[each] = state.h_before + m + da[each]
                    ks[each] = ks_factor(h[each], RAW, TAW)
                    p = DEFAULT_PERC if h[each] > 0 else 0
                    lam[each] = float(row_f["lamda"])
                    h_[each] = h[each] + actual_state.P_1_a - ks[each] * actual_state.et0_a - p
                else:
                    da[each] = max(0, h_[each - 1] - float(row_f["Hp"]))
                    h[each] = h_[each - 1] - da[each]
                    p = DEFAULT_PERC if h[each] > 0 else 0
                    ks[each] = ks_factor(h[each], RAW, TAW)
                    lam[each] = float(row_f["lamda"])
                    h_[each] = h[each] + float(row_a["P_1_a"]) - ks[each] * float(row_a["et0_a"]) - p

                ETC += ks[each] * float(row_a["et0_a"])
                ET += float(row_a["et0_a"])
                P += float(row_a["P_1_a"])
                pp += p

            if abs(m + state.h_before + P) < 0.01:
                r1 = 1
            else:
                r1 = (ETC + pp) / (P + m - h_[range_num - 1] + state.h_before)

            r2 = prod([ks[i] ** lam[i] for i in range(len(ks))])
            if r2 == 0 and action != 0:
                r2 = action / 100
            return r1, r2, r3, r4

        def rice_height(das):
            """种植天数 → 株高估计（mm 量级），供涝渍相对水深用。"""
            return (-0.0059 * das ** 2 + 1.4518 * das + 15.09) * 10

        def water_logging(h_t, h_c, rice_height, alpha, beta):
            """单日涝渍系数：h_t≤Hc → 1；超淹则按 (水深/株高)^β 惩罚，越小越差。"""
            if h_t > h_c:
                return 1 - (alpha * (h_t / rice_height) ** beta) / 50
            return 1

        def alpha_beta_rule(date):
            """生育阶段 → 涝渍敏感参数 (α, β)。抽穗/开花更敏感。"""
            growth_stage = Irrigation.get_growth_stage(date)[0]
            if growth_stage == 'Tillering' or growth_stage == 'Tillering_late':
                alpha, beta = 3, 4
            elif growth_stage == 'Panicle_initiation':
                alpha, beta = 30, 2
            elif growth_stage == 'Flowering':
                alpha, beta = 40, 3
            else:
                alpha, beta = 50, 1
            return alpha, beta

        def cal_r1r3r4(i, actual_state_pd, state_space, state, actual_state, m, d):
            """排侧 / 高水位：7 日前瞻 → r1 效率、r4 涝渍连乘；r2/r3 常为 1。"""
            r2, r3 = 1, 1
            # 与灌侧一致：i 为本季第几天（0-based），勿用 i%Duration（季内本就 < Duration）
            range_num = 7 if (i <= (Duration - 7)) else 7 - (i - (Duration - 7))
            range_num = max(1, min(range_num, len(fc) - i, len(act) - i))

            drainage, RiceHeight, RWL_log, ks, h, h_, h_wl, ks1 = (
                [0.0] * range_num, [0.0] * range_num, [1.0] * range_num, [0.0] * range_num,
                [0.0] * range_num, [0.0] * range_num, [0.0] * range_num, [0.0] * range_num,
            )
            ETC, ET, pp, P, RWL = 0.0, 0.0, 0, 0, 1
            da = [0.0] * range_num

            for each in range(range_num):
                row_f = _fc_at(i + each)
                row_a = _act_at(i + each)
                if each == 0:
                    da[each] = d
                    h[each] = state.h_before + m + da[each]
                    h_wl[each] = h[each]
                    RiceHeight[each] = rice_height(int(Irrigation.get_growth_stage(state.date)[1]))
                    ks[each] = ks_factor(h[each], RAW, TAW)
                    p = DEFAULT_PERC if (h[each] > 0) else 0
                    h_[each] = h[each] + actual_state.P_1_a - ks[each] * actual_state.et0_a - p
                    alpha, beta = alpha_beta_rule(state.date)
                    RWL_log[each] = water_logging(
                        h_[each], float(row_f["Hc"]), RiceHeight[each], alpha, beta
                    )
                else:
                    da[each] = max(0, h_[each - 1] - float(row_f["Hc"]))
                    h[each] = h_[each - 1] - da[each]
                    h_wl[each] = h_wl[each - 1]
                    p = DEFAULT_PERC if h[each] > 0 else 0
                    ks[each] = ks_factor(h[each], RAW, TAW)
                    p1 = DEFAULT_PERC if h[each] > 0 else 0
                    ks1[each] = ks_factor(h[each], RAW, TAW)
                    date_object = datetime.datetime.strptime(str(state.date), "%Y/%m/%d")
                    new_date_object = date_object + datetime.timedelta(days=each)
                    new_date_string = f"{new_date_object.year}/{new_date_object.month}/{new_date_object.day}"
                    RiceHeight[each] = rice_height(
                        int(Irrigation.get_growth_stage(new_date_string)[1]) + each
                    )
                    h_[each] = h[each] + float(row_a["P_1_a"]) - ks[each] * float(row_a["et0_a"]) - p
                    h_wl[each] = (
                        h_wl[each]
                        + float(row_a["P_1_a"])
                        - ks1[each] * float(row_a["et0_a"])
                        - p1
                    )
                    alpha, beta = alpha_beta_rule(new_date_string)
                    RWL_log[each] = water_logging(
                        h_wl[each], float(row_f["Hc"]), RiceHeight[each], alpha, beta
                    )

                ETC += ks[each] * float(row_a["et0_a"])
                ET += float(row_a["et0_a"])
                P += float(row_a["P_1_a"])
                RWL *= RWL_log[each]
                pp += p

            try:
                numerator = ETC + pp
                denominator = P + d - h_[range_num - 1] + state.h_before
                if abs(state.h_before + P + d) < 0.01:
                    r1 = 1
                elif denominator != 0:
                    r1 = numerator / denominator
                else:
                    r1 = action / 100
            except ZeroDivisionError:
                r1 = 0

            # r4：涝渍连乘；推演异常（负系数）时用动作幅度回退
            if any(x < 0 for x in RWL_log):
                r4 = (-action / 100) if action != 0 else 0
            else:
                r4 = RWL
            return r1, r2, r3, r4

        # —— 分支：排水 / 灌溉 / 不动（高水位走排侧推演，否则走灌侧）——
        m, d = cal_md(action)
        if m == 0 and d != 0:
            r1, r2, r3, r4 = cal_r1r3r4(i, actual_state_pd, state_space, state, actual_state, m, d)
        elif m != 0 and d == 0:
            r1, r2, r3, r4 = cal_r1r2(i, actual_state_pd, state_space, state, actual_state, m, d)
        elif m == 0 and d == 0:
            if state.h_before >= state.Hp:
                r1, r2, r3, r4 = cal_r1r3r4(i, actual_state_pd, state_space, state, actual_state, m, d)
            else:
                r1, r2, r3, r4 = cal_r1r2(i, actual_state_pd, state_space, state, actual_state, m, d)

        r0 = cal_r0(i, actual_state_pd, state_space, state, actual_state, m, d)
        r5 = 1 - abs(action) * 0.01  # 大动作略扣分，鼓励小步调节
        r1 = min(r1, 1)
        r = r0 * r1 * r2 * r3 * r4 * r5
        return r, r0, r1, r2, r4

    def correct(self, state, actual_state, action):
        """物理限幅 + 日末水量平衡，返回 (次日 h_before 候选 h_after, 实际执行 mm)。

        灌：水位不得超过 hmax（表上适灌上沿）；排：不得低于 Hp（表上排水目标）。
        亏缺区 (h≤0)：渗漏 p=0；h<-RAW 时实测 ET0 再乘 ks。
        """
        state_ = state.copy()
        actual_action = None

        if action > 0:
            if state_.h_before >= state.hmax:
                actual_action = 0
            elif state_.h_before + action <= state.hmax:
                state_.h_before += action
                actual_action = action
            else:
                actual_action = state.hmax - state_.h_before
                state_.h_before = state.hmax
        elif action < 0:
            if state_.h_before <= state.Hp:
                actual_action = 0
            elif state_.h_before + action <= state.Hp:
                actual_action = state.Hp - state_.h_before
                state_.h_before = state.Hp
            else:
                state_.h_before += action
                actual_action = action
        else:
            actual_action = 0

        sc = _sc()
        # 只用局部 et0，禁止写回 actual_state（避免污染全表，跨 episode 串改）
        et0 = float(actual_state.et0_a)
        if state_.h_before <= 0:
            state_.p = 0
            if state_.h_before < -sc.raw:
                et0 *= ks_factor(state_.h_before, sc.raw, sc.taw)

        # 渗漏须用修正后的 state_.p（亏缺时已置 0），勿减原始 state.p
        state_.h_after = state_.h_before + float(actual_state.P_1_a) - et0 - float(state_.p)
        return state_.h_after, actual_action

    def reset(self, state_space):
        """训练 reset：按「年」均匀随机抽一年，定位 scenario.start，季初 h=hmax。

        返回 (首日 state Series, 当年 actual 子表)。不改写 date 列类型。
        """
        sc = _sc()
        years = state_space["date"].astype(str).str.split("/").str[0].unique()
        year = str(np.random.choice(years))
        formatted_date = _md_to_ymd(year, sc.start)
        matched = state_space[state_space["date"].astype(str) == formatted_date]
        if matched.empty:
            raise ValueError(
                f"找不到季初日 {formatted_date}。请核对 scenario.start={sc.start} 与工作目录 forecast.csv"
            )
        state = matched.squeeze().copy()
        state["h_before"] = state.hmax

        dates = self.actual_state_space_policy["date"].astype(str)
        actual_state_pd = self.actual_state_space_policy[dates.str.split("/").str[0] == year].copy()
        return state, actual_state_pd

    def reset_policy(self, state_space_policy):
        """评估 reset：按年份顺序推进（可复现多年评估），季初 h=hmax。"""
        sc = _sc()
        unique_years = sorted(
            state_space_policy["date"].astype(str).str.split("/").str[0].unique()
        )
        if not hasattr(self, "current_year_idx"):
            self.current_year_idx = 0
        if self.current_year_idx >= len(unique_years):
            self.current_year_idx = 0

        current_year = unique_years[self.current_year_idx]
        self.current_year_idx += 1

        formatted_date = _md_to_ymd(current_year, sc.start)
        matched = state_space_policy[state_space_policy["date"].astype(str) == formatted_date]
        if matched.empty:
            raise ValueError(
                f"找不到季初日 {formatted_date}。请核对 scenario.start={sc.start} 与评估用表"
            )
        state = matched.squeeze().copy()
        state["h_before"] = state.hmax

        dates = self.actual_state_space_policy["date"].astype(str)
        actual_state_pd = self.actual_state_space_policy[
            dates.str.split("/").str[0] == str(current_year)
        ].copy()
        return state, actual_state_pd

    def step(self, actual_state_pd, actual_state_space, state_space, state, action):
        """执行一日：物理限幅 → 水量平衡得次日 h → 算奖励。

        Args:
            actual_state_pd: 当年实测子表（奖励前瞻用）
            actual_state_space: 全日实测（按 date 取当日 P/ET0）
            state_space: 全日 forecast（取次日阶段参数）
            state: 当日状态 Series
            action: 计划灌排 mm（可被 correct 限幅）
        Returns:
            next_state, reward元组(r,r0,r1,r2,r4), done, actual_action
        """
        sc = _sc()
        date_str = str(state.date)
        date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d")
        next_date_obj = date_obj + datetime.timedelta(days=1)
        formatted_date = f"{next_date_obj.year}/{next_date_obj.month}/{next_date_obj.day}"
        md = date_obj.strftime("%m/%d")

        actual_dates = actual_state_space["date"].astype(str)
        actual_state = actual_state_space[actual_dates == date_str].squeeze()
        if isinstance(actual_state, pd.DataFrame) and actual_state.empty:
            raise ValueError(f"actual 缺日 {date_str}")

        i = Irrigation.calculate_days_difference(state, sc.start)

        done = False
        if md == sc.terminal:
            done = True
            next_state = state.copy()
        else:
            next_matched = state_space[state_space["date"].astype(str) == formatted_date]
            if next_matched.empty:
                done = True
                next_state = state.copy()
            else:
                next_state = next_matched.squeeze().copy()

        state_ = next_state.copy()
        state_.h_before, actual_action = self.correct(state, actual_state, action)
        state_.ks = ks_factor(state_.h_before, sc.raw, sc.taw)

        reward = self.reward(actual_state_pd, state_space, state, actual_state, action, i)
        return state_, reward, done, actual_action


