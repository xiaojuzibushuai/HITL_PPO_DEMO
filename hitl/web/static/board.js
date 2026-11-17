/**
 * HITL 单页看板：
 * 训练会话 → 上传/校验(新建) → 训评 → 人工审阅(改动作↔一致性重算) → 行为克隆(BC) → 三方对比
 */
(function () {
  "use strict";

  var ACTION_SPACE = [];
  var runState = {};
  var sessionState = {};
  var expertOverrides = {};
  var expertConfirms = {};
  var expertReasons = {};
  var expertReasonCodes = {};
  var REASON_CODE_OPTIONS = [
    { code: "merge_small_irrigation", label: "合并小额灌水" },
    { code: "wait_forecast_rain", label: "依据预报暂缓灌水" },
    { code: "allow_low_water", label: "允许较低水层" },
    { code: "other", label: "其他" },
  ];
  var lastRows = [];
  /** 当前审阅年（或全部）的完整日序，供水层图；表格可再按分歧日筛 */
  var lastRowsAll = [];
  /** 是否已做过「默认选第一审阅年」；避免用户选「全部」后又被强制回去 */
  var yearFilterAutoDone = false;
  /** boot 下发的规范审阅年（如 2012,2013,2014,2016） */
  var expectedReviewYears = [];
  /** 当前已绑定的训练 / 微调会话 id，用于检测切换并清空回放态 */
  var boundRunId = null;
  var boundSessionId = null;
  var busy = false;
  var pollTimer = null;
  var trainChart = null;
  var actionChart = null;
  var lossValueChart = null;
  var lossPolicyChart = null;
  var yearlyCharts = {};
  var waterChart = null;
  var waterActionChart = null;
  var waterRainChart = null;
  var seasonChart = null;
  var seasonActionChart = null;
  var seasonRainChart = null;
  /** 三方对比区动态 Chart 实例，重绘前统一销毁 */
  var compareCharts = [];
  var ARM_COLORS = {
    rule: "#5b7c99",
    original_ppo: "#1a7a45",
    hitl_ppo: "#c45c26",
  };
  var ROLE_COLORS = {
    bc_train: "#2f6fed",
    bc_val: "#c45c26",
    holdout: "#6b7280",
    other: "#9ca3af",
  };
  /** 三方对比区 GET 缓存（同会话内避免重复打重接口） */
  var compareFetchCache = Object.create(null);
  var compareFetchInflight = Object.create(null);
  function compareFetchJson(url) {
    if (compareFetchCache[url]) {
      return Promise.resolve(compareFetchCache[url]);
    }
    if (compareFetchInflight[url]) return compareFetchInflight[url];
    compareFetchInflight[url] = fetch(url)
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        compareFetchCache[url] = j;
        delete compareFetchInflight[url];
        return j;
      })
      .catch(function (e) {
        delete compareFetchInflight[url];
        throw e;
      });
    return compareFetchInflight[url];
  }
  function clearCompareFetchCache() {
    compareFetchCache = Object.create(null);
    compareFetchInflight = Object.create(null);
  }
  /** 防止 BC/对比连点多次 applySession 并发渲染互相踩 Canvas */
  var comparePanelGen = 0;
  var livePoints = [];
  var lastLogRows = [];
  var lastLossRows = [];
  var lastRunCompare = null;
  var lastYearly = { ppo: [], rule: [] };
  var lastSummary = null;
  /** 入口互斥：'new' | 'load' */
  var entryChoice = "new";
  /** 微调互斥：'load' | 'new' */
  var expertChoice = "load";
  /** 用户点「更换」后强制重选（即使已有绑定） */
  var pickingEntry = false;
  var pickingExpert = false;
  /** 人工审阅表有未一致性重算的本地改动 */
  var reviewDirty = false;

  function $(id) {
    return document.getElementById(id);
  }

  function setMsg(t, isErr) {
    var el = $("msg");
    if (!el) return;
    el.textContent = t || "";
    el.className = "board-msg" + (isErr ? " err" : "");
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(n, d) {
    if (n == null || n === "" || Number.isNaN(Number(n))) return "—";
    return Number(n).toFixed(d == null ? 2 : d);
  }

  /** 动作 mm：缺字段/非数字 → —；0 是合法值（勿 Number(null)→0） */
  function fmtActionMm(v) {
    if (v === null || v === undefined || v === "") return "—";
    var n = Number(v);
    return Number.isFinite(n) ? n.toFixed(0) : "—";
  }

  function fmtPct(n) {
    if (n == null || Number.isNaN(Number(n))) return "—";
    var v = Number(n);
    var sign = v > 0 ? "+" : "";
    return sign + v.toFixed(1) + "%";
  }

  async function post(url, body) {
    var r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    var j = await r.json().catch(function () {
      return {};
    });
    if (!r.ok || j.ok === false) throw new Error(j.error || "HTTP " + r.status);
    return j;
  }

  function setBusy(on) {
    busy = !!on;
    document.body.classList.toggle("is-busy", busy);
    // 校验/上传等 finally 里会 setBusy(false)；须同步刷新训评门闩，
    // 否则 updateTrainGate 在 busy=true 时把「开始运行」锁死，
    // 用户只能靠改「模式」下拉再触发一次才能点（假锁）。
    updateTrainGate();
    updateBcGate();
  }

  function scrollToId(id) {
    var el = $(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function setStepper(step) {
    document.querySelectorAll("#stepper .step").forEach(function (btn) {
      var s = Number(btn.getAttribute("data-step"));
      btn.classList.toggle("on", s === step);
      btn.classList.toggle("done", s < step);
    });
  }

  /** 已训/已评记录：跳过上传与校验 */
  function isLoadedTrainedRun() {
    return !!(runState.run_id && (runState.has_best || runState.has_eval));
  }

  function isNewTrainPipeline() {
    return !!(runState.run_id && !runState.has_best && !runState.has_eval);
  }

  function setHidden(el, hide) {
    if (!el) return;
    el.classList.toggle("section-hidden", !!hide);
  }

  function syncEntryTabs() {
    document.querySelectorAll("#entryTabs .mode-tab").forEach(function (btn) {
      var on = btn.getAttribute("data-entry") === entryChoice;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    });
    setHidden($("entryPaneNew"), entryChoice !== "new");
    setHidden($("entryPaneLoad"), entryChoice !== "load");
  }

  function syncExpertTabs() {
    document.querySelectorAll("#expertTabs .mode-tab").forEach(function (btn) {
      var on = btn.getAttribute("data-expert") === expertChoice;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    });
    setHidden($("expertPaneLoad"), expertChoice !== "load");
    setHidden($("expertPaneNew"), expertChoice !== "new");
  }

  function updateModePill() {
    var pill = $("modePill");
    if (!pill) return;
    pill.classList.remove("is-new", "is-loaded", "is-empty");
    if (!runState.run_id) {
      pill.textContent =
        entryChoice === "load" ? "待加载历史训练会话" : "待新建训练会话";
      pill.classList.add("is-empty");
    } else if (isLoadedTrainedRun()) {
      pill.textContent = "已加载训练会话 · 无需上传/校验";
      pill.classList.add("is-loaded");
    } else {
      pill.textContent = "新建训练会话 · 需上传与校验";
      pill.classList.add("is-new");
    }
  }

  function inferStep() {
    // 0选会话 1上传 2校验 3训评 4人工审阅(含重算) 5行为克隆 6三方
    if (!runState.run_id && !sessionState.session_id) return 0;
    if (sessionState.has_compare) return 6;
    if (sessionState.has_hitl) return 5;
    if (sessionState.has_review || sessionState.session_id) return 4;
    if (runState.has_best || runState.has_eval || runState.train_busy) return 3;
    if (isNewTrainPipeline()) {
      if (runState.data_ok) return 3;
      return runState.has_uploads ? 2 : 1;
    }
    if (runState.run_id) return 3;
    return 0;
  }

  function markReviewDirty(on) {
    reviewDirty = !!on;
    setHidden($("dirtyBanner"), !reviewDirty);
    updateBcGate();
  }

  function pctTxt(v) {
    if (v == null || v === "" || isNaN(Number(v))) return "—";
    return (Number(v) * 100).toFixed(1) + "%";
  }

  function deltaTxt(v) {
    if (v == null || isNaN(Number(v))) return "—";
    var n = Number(v);
    return (n >= 0 ? "+" : "") + n.toFixed(1) + "%";
  }

  function renderBcResultPanel() {
    var el = $("bcResultPanel");
    if (!el) return;
    var d = sessionState.bc_dashboard || {};
    if (!d.trained_ok && !d.has_model) {
      el.classList.add("section-hidden");
      el.innerHTML = "";
      return;
    }
    el.classList.remove("section-hidden");
    var learn = d.learn || {};
    var train = d.train || {};
    var absorb = d.absorb || {};
    var html =
      '<div class="bc-result-grid">' +
      '<div class="bc-result-card">' +
      "<h4>① 学什么</h4>" +
      "<p>导出行 <b>" +
      esc(learn.n_labels_rows) +
      "</b> · 进 BC <b>" +
      esc(learn.n_include_bc) +
      "</b>（改动日 <b>" +
      esc(learn.n_edited) +
      "</b> ×w_edit，未改确认约 <b>" +
      esc(learn.n_keep) +
      "</b> ×1）</p>" +
      '<p class="hint">' +
      esc(learn.explain || "") +
      "</p></div>" +
      '<div class="bc-result-card">' +
      "<h4>② 训成没</h4>" +
      "<p>" +
      (d.trained_ok
        ? '<span class="tag ok">已写出 ppo_hitl.pth</span>'
        : '<span class="tag warn">尚未完成</span>') +
      " · best_epoch " +
      esc(train.best_epoch) +
      "</p>" +
      "<p>训练 CE " +
      esc(train.ce_start != null ? Number(train.ce_start).toFixed(2) : "—") +
      " → " +
      esc(train.ce_end != null ? Number(train.ce_end).toFixed(2) : "—") +
      " · 验证 CE " +
      esc(
        train.val_ce_start != null ? Number(train.val_ce_start).toFixed(2) : "—"
      ) +
      " → " +
      esc(train.val_ce_end != null ? Number(train.val_ce_end).toFixed(2) : "—") +
      "</p></div>" +
      '<div class="bc-result-card">' +
      "<h4>③ 学没学会（吸收率）</h4>" +
      "<p>改动复现 <b>" +
      pctTxt(absorb.edit_reproduce_rate) +
      "</b> · 未改保持 <b>" +
      pctTxt(absorb.confirm_hold_rate_policy) +
      "</b></p>" +
      '<p class="hint">复现=改动日 HITL 仍选专家动作；保持=未改日仍跟原标签。' +
      (absorb.note ? " " + esc(absorb.note) : "") +
      "</p></div>" +
      "</div>";
    el.innerHTML = html;
  }

  function kpiDeltaPct(hitlV, baseV, higherBetter) {
    if (hitlV == null || baseV == null) return null;
    var hv = Number(hitlV);
    var bv = Number(baseV);
    if (isNaN(hv) || isNaN(bv) || Math.abs(bv) < 1e-12) return null;
    var raw = ((hv - bv) / Math.abs(bv)) * 100;
    return higherBetter ? raw : -raw;
  }

  function qualityFromArms(ruleM, ppoM, hitlM) {
    if (!ruleM || !ppoM || !hitlM) return null;
    return {
      vs_rule: {
        rain_utilization_delta_pct: kpiDeltaPct(
          hitlM.rain_utilization,
          ruleM.rain_utilization,
          true
        ),
        managed_water_improve_pct: kpiDeltaPct(
          hitlM.managed_water,
          ruleM.managed_water,
          false
        ),
        yield_factor_delta_pct: kpiDeltaPct(
          hitlM.yield_factor,
          ruleM.yield_factor,
          true
        ),
      },
      vs_original_ppo: {
        rain_utilization_delta_pct: kpiDeltaPct(
          hitlM.rain_utilization,
          ppoM.rain_utilization,
          true
        ),
        managed_water_improve_pct: kpiDeltaPct(
          hitlM.managed_water,
          ppoM.managed_water,
          false
        ),
        yield_factor_delta_pct: kpiDeltaPct(
          hitlM.yield_factor,
          ppoM.yield_factor,
          true
        ),
      },
    };
  }

  /** @param opt {{ year?: string, means?: object, yearRow?: object }} */
  function renderBcQualityHtml(d, opt) {
    d = d || {};
    opt = opt || {};
    var learn = d.learn || {};
    var absorbAll = d.absorb || {};
    var year = (opt.year || "").trim();
    var scopeLabel = year ? year + " 年" : "多年均值";
    var absorb = absorbAll;
    var nEditShow = learn.n_edited;
    var nKeepShow =
      learn.n_keep != null
        ? learn.n_keep
        : absorbAll.n_keep_among_confirmed;
    if (year && absorbAll.by_year && absorbAll.by_year[year]) {
      absorb = absorbAll.by_year[year];
      nEditShow = absorb.n_edited;
      nKeepShow = absorb.n_keep;
    } else if (year) {
      // 留出验证年：审阅未改，无分年吸收样本
      absorb = {
        edit_reproduce_rate: null,
        confirm_hold_rate_policy: null,
        n_edited: 0,
        n_keep: 0,
      };
      nEditShow = 0;
      nKeepShow = 0;
    }

    var q = null;
    if (year && opt.yearRow) {
      q = qualityFromArms(
        opt.yearRow.rule,
        opt.yearRow.original_ppo,
        opt.yearRow.hitl_ppo
      );
    } else if (opt.means) {
      q = qualityFromArms(
        opt.means.rule,
        opt.means.original_ppo,
        opt.means.hitl_ppo
      );
    } else {
      q = d.quality || null;
    }

    var bits = [];
    bits.push(
      '<div class="bc-quality">' +
        "<h3 style='margin:0.75rem 0 0.4rem'>BC 结果摘要 · " +
        esc(scopeLabel) +
        "</h3>" +
        "<p>" +
        (year
          ? "本视图年改动日 " +
            esc(nEditShow) +
            " · 未改进训日 " +
            esc(nKeepShow)
          : "进训样本 " +
            esc(learn.n_include_bc) +
            "（改动 " +
            esc(learn.n_edited) +
            "）") +
        " · 改动复现 " +
        pctTxt(absorb.edit_reproduce_rate) +
        " · 未改保持 " +
        pctTxt(absorb.confirm_hold_rate_policy) +
        "</p>"
    );
    if (year && nEditShow === 0) {
      bits.push(
        '<p class="hint">该年无专家改动（多为留出验证年）。参数更新仍可能改变未标注状态；留出年用于检查这种外溢影响，不能预设 HITL≈原 PPO。吸收率按「本视图年」计。</p>'
      );
    }
    if (q && q.vs_original_ppo) {
      var vo = q.vs_original_ppo;
      var vr = q.vs_rule || {};
      bits.push(
        "<p>相对<strong>原 PPO</strong>（" +
          esc(scopeLabel) +
          "）：雨利用 " +
          deltaTxt(vo.rain_utilization_delta_pct) +
          " · 管理水量改善 " +
          deltaTxt(vo.managed_water_improve_pct) +
          " · 产量系数 " +
          deltaTxt(vo.yield_factor_delta_pct) +
          "</p>"
      );
      bits.push(
        "<p>相对<strong>规则基线</strong>（" +
          esc(scopeLabel) +
          "）：雨利用 " +
          deltaTxt(vr.rain_utilization_delta_pct) +
          " · 管理水量改善 " +
          deltaTxt(vr.managed_water_improve_pct) +
          " · 产量系数 " +
          deltaTxt(vr.yield_factor_delta_pct) +
          "</p>"
      );
    } else {
      bits.push(
        '<p class="hint">跑完「三方对比」后显示相对原 PPO / 规则的改善。</p>'
      );
    }
    bits.push("</div>");
    return bits.join("");
  }

  function updateBcGate() {
    var btn = $("btnBc");
    var hint = $("bcGateHint");
    var hasReview = !!(sessionState.has_review && sessionState.session_id);
    var nEdit = Number(sessionState.n_edit || 0);
    var missing = sessionState.missing_review_years || [];
    var nNeed = Number(sessionState.n_needs_reconfirm || 0);
    // 后端 bc_ready：各规范审阅年均已确认 + 有改动日 + 无待复核；前端再拦脏数据
    var ready = sessionState.bc_ready === true;
    var can = hasReview && !reviewDirty && !busy && ready;
    if (btn) btn.disabled = !can;
    if (hint) {
      if (!hasReview) hint.textContent = "请先完成人工审阅会话绑定";
      else if (reviewDirty)
        hint.textContent = "有未提交改动：请等待自动重算完成";
      else if (missing.length)
        hint.textContent =
          "审阅年未全部定稿（缺 " +
          missing.join(", ") +
          "）：请逐年点「本审阅年定稿」后再 BC";
      else if (nNeed > 0)
        hint.textContent =
          "仍有 " +
          nNeed +
          " 天需重看：请到对应审阅年点「一键认可本审阅年需重看」或逐日认可，清零后再定稿（定稿不能清除需重看）";
      else if (nEdit <= 0)
        hint.textContent =
          "尚未形成改动轨迹：请在审阅年改专家动作、定稿各年后再 BC";
      else if (!ready)
        hint.textContent = "审阅未就绪，暂不可提交行为克隆";
      else
        hint.textContent =
          "已确认审阅年 " +
          (sessionState.confirmed_years || []).join(",") +
          " · 改动日 " +
          nEdit +
          "，可提交行为克隆";
    }
  }

  function formatDateOnly(s) {
    var t = String(s == null ? "" : s).trim();
    if (!t) return "—";
    var m = t.match(/^(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})/);
    if (m) {
      return (
        m[1] +
        "-" +
        String(m[2]).padStart(2, "0") +
        "-" +
        String(m[3]).padStart(2, "0")
      );
    }
    return t.length > 10 ? t.slice(0, 10) : t;
  }

  /** 与后端 _FILE_ROLE_EXACT 对齐；前端兜底，避免只显示「会话产物」 */
  var FILE_ROLES = {
    "forecast.csv": "训练预报气象（日）· 输入表",
    "actual.csv": "训练实测气象（日）· 输入表",
    "forecast_eval.csv": "评估预报气象 · 评估输入",
    "actual_eval.csv": "评估实测气象 · 评估输入",
    "growth_para.csv": "生育期与 hmin/hmax/Hp/Hc/λ 参数",
    "README.txt": "uploads 目录说明",
    "ppo_rice_best.pth": "全局最佳权重 · 评估/审阅用（优先）",
    "ppo_rice_last.pth": "训练终点权重 · 正式评估请用 best",
    "ppo_rice_best_config.json": "best 旁路配置",
    "ppo_rice_last_config.json": "last 旁路配置",
    "compare_summary.json":
      "多年均值对比 · 列 ppo_*/rule_*/delta_pct（雨利用率、灌排、管理水量、产量系数）",
    "policy_compare_daily.csv": "逐日动作对照 · date / a_ppo / a_rule / disagree",
    "policy_output_PPO.csv":
      "PPO 整季逐日 · date / h_before / h_after / a_mm / a_exec / m / d",
    "policy_output_RULE.csv": "规则整季逐日 · 列同 PPO",
    "yearly_metrics_PPO.csv":
      "PPO 分年 KPI · year + rain_utilization / irrigation_total / drainage_total / managed_water / yield_factor",
    "yearly_metrics_RULE.csv": "规则分年 KPI · 列同 PPO",
    "run_manifest.json": "评估产物清单",
    "review.csv":
      "人工审阅表 · 改 a_expert/confirmed；显示决策时可用的 P_cumu7 预报，不显示实测雨",
    "review.json": "人工审阅 JSON",
    "labels.json": "BC 监督标签",
    "ppo_hitl.pth": "行为克隆后 HITL 权重",
    "hitl_result.json": "BC 摘要 · n_labels / n_edited",
    "three_way.json": "三方原始汇总",
    "task_compare.json": "三方任务指标 · 五核心 KPI",
    "absorption.json": "吸收率 · edit_reproduce_rate 等",
  };

  function roleForFile(name) {
    var base = String(name || "").split("/").pop().split("\\").pop();
    if (FILE_ROLES[base]) return FILE_ROLES[base];
    if (/^training_log_.*\.csv$/i.test(base))
      return "逐局训练日志 · episode / return / mean_return / action_count";
    if (/^training_loss_.*\.csv$/i.test(base) && base.indexOf("detail") < 0)
      return "逐局损失 · episode / policy_loss / value_loss / entropy_loss";
    if (/training_loss/i.test(base) && /detail/i.test(base))
      return "训练损失明细（update 级）";
    if (/^training_report_.*\.json$/i.test(base))
      return "训练报告 · training_summary / convergence_analysis";
    if (/^training_results_.*\.png$/i.test(base)) return "训练曲线导出图";
    if (/\/$/.test(name) || /\\$/.test(name)) return "对比臂目录（含该臂 yearly / policy）";
    return "见 docs/场景与输出.md";
  }

  function fileRoleList(detail, fallbackNames) {
    var items =
      detail && detail.length
        ? detail.map(function (it) {
            return {
              name: it.name,
              role: it.role && it.role.indexOf("会话产物") !== 0 ? it.role : roleForFile(it.name),
            };
          })
        : (fallbackNames || []).map(function (n) {
            return { name: n, role: roleForFile(n) };
          });
    if (!items.length) return "<span class='hint'>—</span>";
    return (
      '<ul class="file-role-list">' +
      items
        .map(function (it) {
          return (
            "<li><div class='fname'><code>" +
            esc(it.name) +
            "</code></div><div class='role'>" +
            esc(it.role || "") +
            "</div></li>"
          );
        })
        .join("") +
      "</ul>"
    );
  }

  function metricGuideHtml() {
    return (
      '<div class="metric-guide">' +
      "<h5>看指标 → 打开哪个文件</h5>" +
      "<ul>" +
      "<li><b>学习曲线（回报/动作）</b> → <code>01_train/training_log_*.csv</code>（episode, return, action_count）</li>" +
      "<li><b>训练损失</b> → <code>01_train/training_loss_*.csv</code>（policy_loss, value_loss）</li>" +
      "<li><b>多年均值 KPI / Δ%</b> → <code>02_eval/compare_summary.json</code></li>" +
      "<li><b>分年 KPI</b> → <code>02_eval/yearly_metrics_PPO.csv</code> / <code>_RULE.csv</code></li>" +
      "<li><b>整季水深轨迹</b> → <code>02_eval/policy_output_*.csv</code>（date, h_before, a_exec）</li>" +
      "<li><b>人工改动作</b> → <code>03_review/current/review.csv</code>（a_expert, confirmed）</li>" +
      "<li><b>三方对比</b> → <code>07_metrics/task_compare.json</code></li>" +
      "</ul></div>"
    );
  }

  function renderRunFolder(folder) {
    var body = $("runFolderBody");
    if (!body) return;
    if (!folder || !runState.run_id) {
      body.innerHTML = '<p class="hint">尚未绑定训练会话。</p>';
      return;
    }
    var bits = [
      '<p class="folder-path"><b>路径</b> ' + esc(folder.path) + "</p>",
      metricGuideHtml(),
      '<div class="folder-grid">',
      '<div class="folder-col"><div class="folder-col-h">uploads/ 上传数据</div>' +
        fileRoleList(folder.uploads_detail, folder.uploads) +
        "</div>",
      '<div class="folder-col"><div class="folder-col-h">01_train/ 训练产物</div>' +
        fileRoleList(folder.train_detail, folder.train_files) +
        "</div>",
      '<div class="folder-col"><div class="folder-col-h">02_eval/ 评估产物</div>' +
        fileRoleList(folder.eval_detail, folder.eval_files) +
        "</div>",
      "</div>",
    ];
    if (folder.training_summary) {
      var ts = folder.training_summary;
      bits.push(
        '<p class="folder-stat">训练：局数 ' +
          esc(ts.total_episodes) +
          " · 末段均值 " +
          fmt(ts.final_mean_return, 1) +
          " · 耗时 " +
          fmt(ts.total_time_hours, 2) +
          " h</p>"
      );
    }
    if (folder.n_years != null) {
      bits.push('<p class="folder-stat">评估年数 ' + esc(folder.n_years) + "</p>");
    }
    body.innerHTML = bits.join("");
  }

  function renderSessFolder(folder) {
    var html;
    if (!folder || !sessionState.session_id) {
      html = '<p class="hint">尚未绑定微调会话。</p>';
    } else {
      html =
        '<p class="folder-path"><b>路径</b> ' +
        esc(folder.path) +
        "</p>" +
        metricGuideHtml() +
        '<div class="folder-grid">' +
        '<div class="folder-col"><div class="folder-col-h">03_review · 人工审阅</div>' +
        fileRoleList(folder.review_detail, folder.review_files) +
        "</div>" +
        '<div class="folder-col"><div class="folder-col-h">04_bc_export · 标签</div>' +
        fileRoleList(folder.bc_export_detail, folder.bc_export) +
        "</div>" +
        '<div class="folder-col"><div class="folder-col-h">05_bc · 行为克隆</div>' +
        fileRoleList(folder.bc_detail, folder.bc_files) +
        "</div>" +
        '<div class="folder-col"><div class="folder-col-h">06_compare · 三方</div>' +
        fileRoleList(folder.compare_detail, folder.compare_files) +
        "</div>" +
        '<div class="folder-col"><div class="folder-col-h">07_metrics · 指标</div>' +
        fileRoleList(folder.metrics_detail, folder.metrics_files) +
        "</div>" +
        "</div>";
    }
    if ($("sessFolderInlineBody")) $("sessFolderInlineBody").innerHTML = html;
  }

  function updateTrainGate() {
    var btn = $("btnStartTrain");
    var hint = $("trainGateHint");
    var mode = ($("trainMode") && $("trainMode").value) || "auto";
    var loaded = isLoadedTrainedRun();
    var dv = runState.data_validate || {};
    // 上次若只验了评估集（need_train=false），切回训练/自动模式须重验训表
    var trainTablesOk = !!runState.data_ok && dv.need_train !== false;
    var can = !!runState.run_id;
    if (mode === "eval_only") can = can && !!runState.has_best;
    else can = can && trainTablesOk;
    if (btn) btn.disabled = !can || !!busy;
    if ($("btnGotoExpert")) $("btnGotoExpert").disabled = !runState.has_best;
    if (hint) {
      if (!runState.run_id) hint.textContent = "请先选择 / 新建训练会话";
      else if (loaded && !trainTablesOk && mode !== "eval_only")
        hint.textContent =
          "已加载训练会话：可回放结果并进入人工审阅；若要重训请先校验数据，或改选「仅整季推理」";
      else if (!(runState.uploads || []).length && mode !== "eval_only")
        hint.textContent =
          "uploads 为空：请「保存上传文件」或点「使用默认情景表」，再做数据集校验";
      else if (!runState.data_ok && mode !== "eval_only")
        hint.textContent = "请先点「② 数据集校验」；通过后「开始运行」才会可用";
      else if (runState.data_ok && dv.need_train === false && mode !== "eval_only")
        hint.textContent =
          "当前校验仅为评估集：请点「② 数据集校验」重验训表，或改选「仅整季推理」";
      else if (mode === "eval_only" && !runState.has_best)
        hint.textContent = "仅整季推理需要已有 best 或上传 .pth";
      else if (loaded) hint.textContent = "已加载训练会话：可回放曲线或进入人工审阅";
      else hint.textContent = "校验已通过，可以开始运行";
    }
    if ($("modelChip")) {
      $("modelChip").textContent = runState.has_best
        ? "已有权重：best.pth"
        : "尚未放入已训模型。";
    }
  }

  function syncBoardVisibility() {
    syncEntryTabs();
    syncExpertTabs();
    updateModePill();

    var hasRun = !!runState.run_id;
    var hasSess = !!sessionState.session_id;
    var loaded = isLoadedTrainedRun();
    var isNewPath = hasRun && !loaded;
    var hasReview = !!(sessionState.has_review && hasSess);
    var reviewerBlind =
      hasReview && (!sessionState.has_hitl || !sessionState.bc_ready) && !pickingExpert;

    // 入口：绑定后收起选项卡；点「更换」可重选。互斥只显示当前面板
    var showEntryPick = !hasRun || pickingEntry;
    setHidden($("entryTabs"), !showEntryPick);
    setHidden($("entryPaneNew"), !showEntryPick || entryChoice !== "new");
    setHidden($("entryPaneLoad"), !showEntryPick || entryChoice !== "load");
    setHidden($("runFolderBox"), !hasRun || pickingEntry);

    // 管线：有绑定且未在重选入口时显示
    var showPipe = (hasRun || hasSess) && !pickingEntry;
    setHidden($("sec-pipeline"), !showPipe);

    // 上传仅新建路径
    setHidden($("card-upload"), !(showPipe && hasRun && isNewPath));
    setHidden($("card-train"), !(showPipe && hasRun && !reviewerBlind));

    // 专家区
    var showExpert = showPipe && (hasRun || hasSess);
    setHidden($("card-expert"), !showExpert);
    var showExpertPick = showExpert && (!hasSess || pickingExpert);
    setHidden($("expertGateBlock"), !showExpertPick);
    setHidden($("sessFolderInline"), !(showExpert && hasSess && !pickingExpert));
    setHidden($("expertMetrics"), !(showExpert && hasSess && !pickingExpert));
    setHidden($("reviewPane"), !(showExpert && hasReview && !pickingExpert));

    // 行为克隆 / 三方：有微调会话且未在重选
    setHidden(
      $("card-bc"),
      !(showPipe && hasSess && !pickingExpert && !(reviewerBlind && sessionState.has_hitl))
    );
    setHidden($("card-eval"), !(showPipe && hasSess && !pickingExpert && !reviewerBlind));

    var flow = $("flowHint");
    if (flow) {
      if (!hasRun) {
        flow.textContent =
          entryChoice === "load"
            ? "请选择并加载历史训练会话 → 训评结果 → 人工审阅 → …"
            : "请新建训练会话 → 上传/校验 → 训评 → 人工审阅 → …";
      } else if (loaded) {
        flow.textContent =
          "已加载训练会话 → 训评结果 → 人工审阅（改动作→一致性重算）→ 行为克隆(BC) → 三方对比";
      } else {
        flow.textContent =
          "新建训练会话 → 上传/校验 → 训评结果 → 人工审阅（改动作→一致性重算）→ 行为克隆(BC) → 三方对比";
      }
    }

    var hint = $("trainHint");
    if (hint) {
      hint.textContent = loaded
        ? "当前为历史训练会话回放：上传/校验已隐藏。可直接看曲线，或「仅整季推理」/ 进入人工审阅。"
        : "新建路径：须先完成上传与校验，再开始训练或整季推理。";
    }

    document.querySelectorAll(".step[data-new-only]").forEach(function (btn) {
      var skip = !hasRun || loaded;
      btn.classList.toggle("is-skipped", skip);
      btn.disabled = skip;
      btn.title = skip ? "当前路径不适用（已置灰）" : "";
    });

    updateTrainGate();
    updateBcGate();
  }

  function updateLocks() {
    syncBoardVisibility();
  }

  function resetUploadFileInputs() {
    [
      "upForecast",
      "upActual",
      "upForecastEval",
      "upActualEval",
      "upGrowth",
      "upModel",
    ].forEach(function (id) {
      var el = $(id);
      if (el) el.value = "";
    });
  }

  function renderUploadSourceBanner(st) {
    var el = $("uploadSourceBanner");
    if (!el) return;
    if (!st || !st.run_id) {
      el.textContent = "";
      return;
    }
    var src = st.uploads_source || "";
    var files = (st.uploads || []).join(", ") || "（空）";
    if (src === "scenario_snapshot") {
      var snap = st.uploads_snapshot || {};
      el.innerHTML =
        "<b>当前 uploads 来源：默认情景快照</b>" +
        (snap.scenario ? "（" + esc(snap.scenario) + "）" : "") +
        " · 已有 " +
        esc(files) +
        "。须点「② 数据集校验」通过后才能训练；要换数据请「选择文件」→「保存上传文件」再校验。";
    } else if ((st.uploads || []).length) {
      el.innerHTML =
        "<b>当前 uploads 来源：本 run 目录已有表</b> · " +
        esc(files) +
        "。须校验通过后才能「开始运行」。";
    } else {
      el.innerHTML =
        "<b>uploads 为空</b>：请「保存上传文件」，或点「使用默认情景表」填入情景 csv，再点「② 数据集校验」。";
    }
  }

  function applyRun(st, opt) {
    if (!st) return;
    opt = opt || {};
    var nextId = st.run_id || null;
    var switched = opt.reset || nextId !== boundRunId;
    // 切换 / 强制重置：先清空训评回放，再绑定新状态
    if (switched) {
      clearRunPlayback();
      resetUploadFileInputs();
      boundRunId = nextId;
    }
    runState = st;
    if (st.run_id) {
      if (st.has_best || st.has_eval) entryChoice = "load";
      else entryChoice = "new";
    }
    if ($("runLabel")) $("runLabel").textContent = st.run_id || "未选择";
    var chips = $("uploadChips");
    if (chips) {
      chips.innerHTML = (st.uploads || [])
        .map(function (f) {
          return "<code>" + esc(f) + "</code>";
        })
        .join("");
    }
    if (st.data_validate) showValidate(st.data_validate);
    else if ($("validateBox")) {
      $("validateBox").className = "validate-box";
      $("validateBox").innerHTML =
        '<p class="hint" style="margin:0">尚未校验。将校验本 run 的 <code>uploads/</code> 磁盘文件（见上方来源说明）。</p>';
    }
    renderUploadSourceBanner(st);
    renderRunFolder(st.folder);
    updateTrainGate();
    updateLocks();
    setStepper(inferStep());
  }

  function applySession(st, opt) {
    if (!st) return;
    opt = opt || {};
    var nextId = st.session_id || null;
    if (opt.reset || nextId !== boundSessionId) {
      clearSessionPlayback();
      clearCompareFetchCache();
      boundSessionId = nextId;
    }
    sessionState = st;
    if (st.session_id) {
      // 有现成审阅视为「加载」路径；刚新建也会落到有 sess
      if (!expertChoice) expertChoice = "load";
    }
    if ($("sessLabel")) $("sessLabel").textContent = st.session_id || "未选择";
    if ($("mDays")) $("mDays").textContent = st.n_days || 0;
    if ($("mPrefer")) $("mPrefer").textContent = st.n_prefer || 0;
    if ($("mConf")) $("mConf").textContent = st.n_confirmed || 0;
    if ($("mEdit")) $("mEdit").textContent = st.n_edit || 0;
    if ($("bcRationale"))
      $("bcRationale").textContent =
        st.bc_rationale || "完成人工审阅并形成改动轨迹后，可在此提交行为克隆。";
    if (st.expected_review_years && st.expected_review_years.length) {
      expectedReviewYears = st.expected_review_years.map(String);
      if ($("reviewYearsLabel"))
        $("reviewYearsLabel").textContent = expectedReviewYears.join(",");
    }
    var hasReview = !!st.has_review;
    if (hasReview) loadRows();
    else {
      lastRows = [];
      lastRowsAll = [];
      updateWaterChart([]);
      if ($("reviewTbody"))
        $("reviewTbody").innerHTML =
          '<tr><td colspan="14" class="empty">请先生成或加载人工审阅表</td></tr>';
    }
    renderSessFolder(st.folder);
    renderBcResultPanel();
    if (!opt.skipComparePanel) renderComparePanel();
    markReviewDirty(false);
    updateBcGate();
    updateLocks();
    setStepper(inferStep());
  }

  function showValidate(v) {
    var box = $("validateBox");
    if (!box || !v) return;
    if (v.ok) {
      box.className = "validate-box ok";
      var srcNote =
        v.source === "uploads"
          ? "磁盘 uploads/"
          : v.source === "scenario"
            ? "情景默认表"
            : esc(v.source || "uploads");
      box.innerHTML =
        '<p class="ok-line">校验通过（' +
        srcNote +
        "）· " +
        esc(v.start) +
        "～" +
        esc(v.terminal) +
        "</p>" +
        '<p class="hint" style="margin:0.35rem 0 0">依据是本 run 目录里的表。新建默认不拷情景；可用「使用默认情景表」或手传后再校验。</p>';
    } else {
      box.className = "validate-box bad";
      var errs = (v.errors || [])
        .map(function (e) {
          return "<li>" + esc(e) + "</li>";
        })
        .join("");
      box.innerHTML = '<p class="bad-line">校验未通过</p><ul>' + errs + "</ul>";
    }
  }

  function ensureTrainChart() {
    if (trainChart || !$("trainChart") || typeof Chart === "undefined") return;
    trainChart = new Chart($("trainChart"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "原始回报",
            data: [],
            borderColor: "rgba(70, 95, 82, 0.45)",
            borderWidth: 1.15,
            pointRadius: 0,
            tension: 0.12,
            order: 2,
          },
          {
            label: "平滑 100 局",
            data: [],
            borderColor: "#1f4d36",
            borderWidth: 2.4,
            pointRadius: 0,
            tension: 0.22,
            order: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: { boxWidth: 14, font: { size: 13 }, color: "#2a3a32" },
          },
        },
        scales: {
          x: {
            ticks: { font: { size: 12 }, maxTicksLimit: 10, color: "#6a7a72" },
            grid: { display: false },
            title: { display: true, text: "Episode", font: { size: 12 }, color: "#6a7a72" },
          },
          y: {
            ticks: { font: { size: 13 }, color: "#6a7a72" },
            grid: { color: "#eef2ef" },
            title: { display: true, text: "回报", font: { size: 12 }, color: "#6a7a72" },
          },
        },
      },
    });
  }

  function updateTrainChart(rows, lossRows) {
    updateProcessCharts(rows, lossRows);
  }

  function resetTrainMetricsUi() {
    if ($("tmEp")) $("tmEp").textContent = "—";
    if ($("tmRet")) $("tmRet").textContent = "—";
    if ($("tmMean")) $("tmMean").textContent = "—";
    if ($("tmBest")) $("tmBest").textContent = "—";
    if ($("tmStatus")) {
      $("tmStatus").textContent = "待机";
      $("tmStatus").className = "";
    }
    if ($("trainStatusLine")) $("trainStatusLine").textContent = "未开始";
    if ($("convHint")) $("convHint").textContent = "收敛摘要将显示在此。";
  }

  function clearEvalCharts() {
    Object.keys(yearlyCharts).forEach(function (k) {
      var ch = yearlyCharts[k];
      if (!ch) return;
      ch.data.labels = [];
      ch.data.datasets = [];
      ch.update("none");
    });
    setHidden($("seasonChartWrap"), true);
    setHidden($("yearlyChartWrap"), false);
    if ($("evalYearFilter")) $("evalYearFilter").innerHTML = '<option value="">多年均值（默认）</option>';
    [seasonChart, seasonActionChart, seasonRainChart].forEach(function (ch) {
      if (!ch) return;
      ch.data.labels = [];
      ch.data.datasets = [];
      ch.update("none");
    });
  }

  /** 切换/新建训练会话时清空训评回放态，避免残留上一 run 的曲线与摘要 */
  function clearRunPlayback() {
    stopPoll();
    livePoints = [];
    lastLogRows = [];
    lastLossRows = [];
    lastSummary = null;
    lastRunCompare = null;
    lastYearly = { ppo: [], rule: [] };
    resetTrainMetricsUi();
    updateProcessCharts([], []);
    clearEvalCharts();
    renderSummary(null);
    if ($("summaryBody"))
      $("summaryBody").innerHTML =
        '<p class="hint">训评完成后显示；加载历史训练会话后可回放。</p>';
    if ($("runFolderBody"))
      $("runFolderBody").innerHTML = '<p class="hint">绑定训练会话后显示文件夹摘要。</p>';
  }

  function clearChartData(ch) {
    if (!ch) return;
    ch.data.labels = [];
    ch.data.datasets = [];
    ch.update("none");
  }

  /** 切换/新建 HITL 微调会话时清空审阅表、本地改动、水层图与三方区 */
  function clearSessionPlayback() {
    expertOverrides = {};
    expertConfirms = {};
    expertReasons = {};
    expertReasonCodes = {};
    lastRows = [];
    lastRowsAll = [];
    yearFilterAutoDone = false;
    markReviewDirty(false);
    clearChartData(waterChart);
    clearChartData(waterActionChart);
    clearChartData(waterRainChart);
    if ($("yearFilter"))
      $("yearFilter").innerHTML = '<option value="">全部审阅年</option>';
    if ($("yearGapHint")) {
      $("yearGapHint").textContent = "";
      setHidden($("yearGapHint"), true);
    }
    if ($("reviewTbody"))
      $("reviewTbody").innerHTML =
        '<tr><td colspan="14" class="empty">请先生成或加载人工审阅表</td></tr>';
    if ($("mDays")) $("mDays").textContent = "0";
    if ($("mPrefer")) $("mPrefer").textContent = "0";
    if ($("mConf")) $("mConf").textContent = "0";
    if ($("mEdit")) $("mEdit").textContent = "0";
    if ($("resyncStatus"))
      $("resyncStatus").textContent =
        "改动作后务必重算，再继续审阅或进入行为克隆。";
    if ($("sessFolderInlineBody"))
      $("sessFolderInlineBody").innerHTML =
        '<p class="hint">加载或新建后显示。</p>';
    if ($("compareBox"))
      $("compareBox").innerHTML =
        '<p class="hint">三方对比完成后显示指标表。</p>';
    if ($("compareYearFilter"))
      $("compareYearFilter").innerHTML =
        '<option value="">多年均值（默认）</option>';
    if ($("bcRationale"))
      $("bcRationale").textContent =
        "完成人工审阅并形成改动轨迹后，可在此提交行为克隆。";
    if ($("sessLabel")) $("sessLabel").textContent = "未选择";
    if ($("yearFilter"))
      $("yearFilter").innerHTML = '<option value="">（未加载）</option>';
    updateBcGate();
  }

  function trainStatusLabel(prog) {
    if (!prog) return "待机";
    var st = String(prog.status || prog.phase || "").toLowerCase();
    if (prog.done === true || st === "done") return "完成";
    if (st === "error" || prog.error) return "失败";
    if (st === "evaluating" || st === "eval") return "评估";
    if (st === "training" || st === "train") return "训练";
    if (st === "idle" || !st) return "待机";
    return "进行中";
  }

  function updateTrainMetrics(prog) {
    if (!prog) {
      resetTrainMetricsUi();
      return;
    }
    if ($("tmEp")) $("tmEp").textContent = prog.episode != null ? prog.episode : "—";
    if ($("tmRet")) $("tmRet").textContent = fmt(prog.return, 1);
    if ($("tmMean")) $("tmMean").textContent = fmt(prog.mean_return, 1);
    if ($("tmBest"))
      $("tmBest").textContent = fmt(
        prog.best_mean_return != null ? prog.best_mean_return : prog.best_mean,
        1
      );
    var label = trainStatusLabel(prog);
    if ($("tmStatus")) $("tmStatus").textContent = label;
    if ($("trainStatusLine")) {
      $("trainStatusLine").textContent =
        prog.message ||
        (prog.error ? String(prog.error) : "") ||
        label ||
        "进行中…";
    }
  }

  function deltaClass(v) {
    if (v == null || Number.isNaN(Number(v))) return "flat";
    var n = Number(v);
    if (Math.abs(n) < 0.05) return "flat";
    return n > 0 ? "up" : "down";
  }

  /** PPO vs 固定规则基线摘要表 */
  function kpiCard(label, ppo, rule, delta, unit) {
    unit = unit || "";
    return (
      "<div><b>" +
      (typeof ppo === "number" ? fmt(ppo, unit === "%" ? 1 : 2) : esc(ppo)) +
      (unit === "%" && typeof ppo === "number" ? "%" : "") +
      "</b><span>" +
      esc(label) +
      "</span>" +
      '<span class="' +
      deltaClass(delta) +
      '" style="display:block;margin-top:0.15rem">' +
      "基线 " +
      (typeof rule === "number" ? fmt(rule, unit === "%" ? 1 : 2) : esc(rule)) +
      (unit === "%" && typeof rule === "number" ? "%" : "") +
      " · Δ " +
      fmtPct(delta) +
      "</span></div>"
    );
  }

  function renderRunCompareTable(cs, opt) {
    if (!cs) return "";
    opt = opt || {};
    var d = cs.delta_pct || {};
    var title = opt.title || "评估集多年均值 · vs 基线(规则)";
    var cards =
      '<div class="metric-cards">' +
      kpiCard("雨利用率", (cs.ppo_rain_utilization || 0) * 100, (cs.rule_rain_utilization || 0) * 100, d.rain_utilization, "%") +
      kpiCard("灌总量 mm", cs.ppo_irrigation_total, cs.rule_irrigation_total, d.irrigation_total) +
      kpiCard("排总量 mm", cs.ppo_drainage_total, cs.rule_drainage_total, d.drainage_total) +
      kpiCard("管理水量", cs.ppo_managed_water, cs.rule_managed_water, d.managed_water) +
      kpiCard("产量水分系数", cs.ppo_yield_factor, cs.rule_yield_factor, d.yield_factor) +
      kpiCard("灌次数", cs.ppo_irrigation_times, cs.rule_irrigation_times, d.irrigation_times) +
      kpiCard("排次数", cs.ppo_drainage_times, cs.rule_drainage_times, d.drainage_times) +
      "</div>";
    // 看板只展示论文五核心 + 灌排次数；胁迫/超Hc 等扩展项仍写在 yearly CSV，不占主表
    var rows = [
      ["雨利用率", cs.ppo_rain_utilization, cs.rule_rain_utilization, d.rain_utilization],
      ["灌总量", cs.ppo_irrigation_total, cs.rule_irrigation_total, d.irrigation_total],
      ["排总量", cs.ppo_drainage_total, cs.rule_drainage_total, d.drainage_total],
      ["管理水量", cs.ppo_managed_water, cs.rule_managed_water, d.managed_water],
      ["产量水分系数", cs.ppo_yield_factor, cs.rule_yield_factor, d.yield_factor],
      ["灌次数", cs.ppo_irrigation_times, cs.rule_irrigation_times, d.irrigation_times],
      ["排次数", cs.ppo_drainage_times, cs.rule_drainage_times, d.drainage_times],
    ];
    var body = rows
      .map(function (r) {
        return (
          "<tr><td>" +
          esc(r[0]) +
          "</td><td>" +
          fmt(r[1], 3) +
          "</td><td>" +
          fmt(r[2], 3) +
          '</td><td class="' +
          deltaClass(r[3]) +
          '">' +
          fmtPct(r[3]) +
          "</td></tr>"
        );
      })
      .join("");
    var disagree = cs.disagree
      ? " · 分歧率 " + fmt((cs.disagree.disagree_rate || 0) * 100, 1) + "%"
      : "";
    return (
      cards +
      '<p class="hint">' +
      esc(title) +
      " · n=" +
      esc(cs.n_years_ppo || "—") +
      disagree +
      " · Δ% = 相对固定规则基线（绿=更好）</p>" +
      '<table class="compare-table"><thead><tr><th>指标</th><th>PPO</th><th>基线(规则)</th><th>Δ% vs 基线</th></tr></thead><tbody>' +
      body +
      "</tbody></table>"
    );
  }

  function yearlyPairToCompare(ppoRow, ruleRow) {
    if (!ppoRow && !ruleRow) return null;
    ppoRow = ppoRow || {};
    ruleRow = ruleRow || {};
    function dlt(a, b, lowerBetter) {
      if (a == null || b == null || Math.abs(Number(b)) < 1e-12) return null;
      var pct = ((Number(a) - Number(b)) / Math.abs(Number(b))) * 100;
      return lowerBetter ? -pct : pct;
    }
    return {
      n_years_ppo: 1,
      ppo_rain_utilization: ppoRow.rain_utilization,
      rule_rain_utilization: ruleRow.rain_utilization,
      ppo_irrigation_total: ppoRow.irrigation_total,
      rule_irrigation_total: ruleRow.irrigation_total,
      ppo_drainage_total: ppoRow.drainage_total,
      rule_drainage_total: ruleRow.drainage_total,
      ppo_managed_water: ppoRow.managed_water,
      rule_managed_water: ruleRow.managed_water,
      ppo_yield_factor: ppoRow.yield_factor,
      rule_yield_factor: ruleRow.yield_factor,
      ppo_irrigation_times: ppoRow.irrigation_times,
      rule_irrigation_times: ruleRow.irrigation_times,
      ppo_drainage_times: ppoRow.drainage_times,
      rule_drainage_times: ruleRow.drainage_times,
      delta_pct: {
        rain_utilization: dlt(ppoRow.rain_utilization, ruleRow.rain_utilization, false),
        irrigation_total: dlt(ppoRow.irrigation_total, ruleRow.irrigation_total, true),
        drainage_total: dlt(ppoRow.drainage_total, ruleRow.drainage_total, true),
        managed_water: dlt(ppoRow.managed_water, ruleRow.managed_water, true),
        yield_factor: dlt(ppoRow.yield_factor, ruleRow.yield_factor, false),
        irrigation_times: dlt(ppoRow.irrigation_times, ruleRow.irrigation_times, true),
        drainage_times: dlt(ppoRow.drainage_times, ruleRow.drainage_times, true),
      },
    };
  }

  function movingAvg(arr, w) {
    if (!w || w <= 1) return arr.slice();
    var out = [];
    for (var i = 0; i < arr.length; i++) {
      var a = Math.max(0, i - w + 1);
      var s = 0;
      var n = 0;
      for (var j = a; j <= i; j++) {
        if (arr[j] != null && !Number.isNaN(arr[j])) {
          s += arr[j];
          n++;
        }
      }
      out.push(n ? s / n : null);
    }
    return out;
  }

  function ensureLineChart(existing, canvasId, datasets) {
    if (existing || !$(canvasId) || typeof Chart === "undefined") return existing;
    return new Chart($(canvasId), {
      type: "line",
      data: { labels: [], datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 18, font: { size: 16 } } },
        },
        scales: {
          x: { ticks: { font: { size: 15 }, maxTicksLimit: 12 }, grid: { color: "#eef2ef" } },
          y: { ticks: { font: { size: 16 } }, grid: { color: "#e8eee9" } },
        },
      },
    });
  }

  function updateProcessCharts(rows, lossRows) {
    // 显式传入（含 []）则覆盖；勿用 || 否则空数组也会落到旧缓存
    if (rows != null) lastLogRows = rows;
    if (lossRows != null) lastLossRows = lossRows;
    lastLogRows = lastLogRows || [];
    lastLossRows = lastLossRows || [];
    var w = Number(($("smoothWindow") && $("smoothWindow").value) || 0);
    var labels = lastLogRows.map(function (r) {
      return r.episode;
    });
    var rets = lastLogRows.map(function (r) {
      return r.return;
    });
    var means = lastLogRows.map(function (r) {
      return r.mean_return;
    });
    var acts = lastLogRows.map(function (r) {
      return r.action_count;
    });
    var smooth = movingAvg(rets, w);

    ensureTrainChart();
    if (trainChart) {
      trainChart.data.labels = labels;
      trainChart.data.datasets = [
        {
          label: "原始回报",
          data: rets,
          borderColor: "rgba(70, 95, 82, 0.45)",
          borderWidth: 1.15,
          pointRadius: 0,
          tension: 0.12,
          order: 2,
        },
        {
          label: w > 0 ? "平滑 " + w + " 局" : "局均回报",
          data: w > 0 ? smooth : means,
          borderColor: "#1f4d36",
          borderWidth: 2.4,
          pointRadius: 0,
          tension: 0.22,
          order: 1,
        },
      ];
      trainChart.options.plugins.legend.display = true;
      trainChart.options.plugins.legend.labels = {
        boxWidth: 14,
        font: { size: 13 },
        color: "#2a3a32",
      };
      trainChart.update("none");
    }

    actionChart = ensureLineChart(actionChart, "actionChart", [
      {
        label: "非零动作数",
        data: [],
        borderColor: "#3d7a5a",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        backgroundColor: "rgba(61,122,90,0.12)",
      },
    ]);
    if (actionChart) {
      actionChart.data.labels = labels;
      actionChart.data.datasets[0].data = acts;
      actionChart.update("none");
    }

    lossValueChart = ensureLineChart(lossValueChart, "lossValueChart", [
      {
        label: "value",
        data: [],
        borderColor: "#9a5b12",
        borderWidth: 1.6,
        pointRadius: 0,
        tension: 0.15,
      },
    ]);
    lossPolicyChart = ensureLineChart(lossPolicyChart, "lossPolicyChart", [
      {
        label: "policy",
        data: [],
        borderColor: "#2d5a42",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
      },
      {
        label: "entropy",
        data: [],
        borderColor: "#6b7c74",
        borderWidth: 1.4,
        pointRadius: 0,
        tension: 0.15,
        borderDash: [4, 3],
      },
    ]);
    var lossLabels = lastLossRows.map(function (r) {
      return r.episode;
    });
    if (lossValueChart) {
      lossValueChart.data.labels = lossLabels;
      lossValueChart.data.datasets[0].data = lastLossRows.map(function (r) {
        return r.value_loss;
      });
      lossValueChart.update("none");
    }
    if (lossPolicyChart) {
      lossPolicyChart.data.labels = lossLabels;
      lossPolicyChart.data.datasets[0].data = lastLossRows.map(function (r) {
        return r.policy_loss;
      });
      lossPolicyChart.data.datasets[1].data = lastLossRows.map(function (r) {
        return r.entropy_loss;
      });
      lossPolicyChart.update("none");
    }
  }

  var YEARLY_METRICS_ORDER = [
    { key: "rain_utilization", label: "雨利用率", unit: "%", scale: 100 },
    { key: "irrigation_total", label: "灌总量", unit: "mm", scale: 1 },
    { key: "drainage_total", label: "排总量", unit: "mm", scale: 1 },
    { key: "managed_water", label: "管理水量", unit: "mm", scale: 1 },
    { key: "yield_factor", label: "产量水分系数", unit: "", scale: 1 },
  ];

  function updateYearlyChart(ppoRows, ruleRows) {
    if (ppoRows != null || ruleRows != null) {
      lastYearly = { ppo: ppoRows || [], rule: ruleRows || [] };
    }
    if (typeof Chart === "undefined") return;
    var years = (lastYearly.ppo.length ? lastYearly.ppo : lastYearly.rule).map(function (r) {
      return r.year;
    });
    function col(rows, key) {
      var m = {};
      (rows || []).forEach(function (r) {
        m[r.year] = r[key];
      });
      return years.map(function (y) {
        return m[y] != null ? Number(m[y]) : null;
      });
    }
    YEARLY_METRICS_ORDER.forEach(function (meta) {
      var canvasId = "yearlyChart_" + meta.key;
      if (!$(canvasId)) return;
      if (!yearlyCharts[meta.key]) {
        yearlyCharts[meta.key] = new Chart($(canvasId), {
          type: "bar",
          data: { labels: [], datasets: [] },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: "top",
                labels: { boxWidth: 12, font: { size: 12 } },
              },
            },
            scales: {
              x: { ticks: { font: { size: 12 } }, grid: { display: false } },
              y: { ticks: { font: { size: 12 } }, grid: { color: "#e8eee9" } },
            },
          },
        });
      }
      var ch = yearlyCharts[meta.key];
      var scale = meta.scale || 1;
      var unitBit = meta.unit ? " " + meta.unit : "";
      ch.data.labels = years;
      ch.data.datasets = [
        {
          label: "PPO" + unitBit,
          data: col(lastYearly.ppo, meta.key).map(function (v) {
            return v == null ? null : v * scale;
          }),
          backgroundColor: "#2d5a42",
        },
        {
          label: "基线" + unitBit,
          data: col(lastYearly.rule, meta.key).map(function (v) {
            return v == null ? null : v * scale;
          }),
          backgroundColor: "#8fb59c",
        },
      ];
      ch.update("none");
    });
  }

  function fillEvalYearSelect(years) {
    var sel = $("evalYearFilter");
    if (!sel) return;
    var cur = sel.value;
    sel.innerHTML =
      '<option value="">多年均值（默认）</option>' +
      (years || [])
        .map(function (y) {
          return '<option value="' + esc(y) + '">' + esc(y) + " 整季</option>";
        })
        .join("");
    if (cur && years && years.indexOf(String(cur)) >= 0) sel.value = cur;
  }

  function findYearlyRow(rows, year) {
    var y = String(year);
    for (var i = 0; i < (rows || []).length; i++) {
      if (String(rows[i].year) === y) return rows[i];
    }
    return null;
  }

  function actionMm(r) {
    var a = r.a_exec != null ? r.a_exec : r.a_mm != null ? r.a_mm : r.a_cmd;
    var n = Number(a);
    return Number.isFinite(n) ? n : 0;
  }

  function rainMm(r) {
    // 训评回放中的事后实测雨量；审阅表使用单独的 forecastRainMm。
    var v =
      r.rain != null
        ? r.rain
        : r.P_1_a != null
          ? r.P_1_a
          : r.P_actual_display;
    var n = Number(v);
    if (!Number.isFinite(n) || String(v).trim() === "?") return 0;
    return n;
  }

  function forecastRainMm(r) {
    var v = r.forecast_rain_7d != null ? r.forecast_rain_7d : r.P_cumu7;
    if (v == null || v === "" || String(v).trim() === "?") return null;
    var n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function timeAxisOptions() {
    return {
      ticks: {
        font: { size: 13 },
        maxTicksLimit: 14,
        callback: function (val) {
          var lab = this.getLabelForValue(val);
          var d = formatDateOnly(lab);
          return d.length >= 10 ? d.slice(5) : d;
        },
      },
      grid: { color: "#eef2ef" },
    };
  }

  var AXIS_LEFT_W = 52;
  var AXIS_RIGHT_W = 56;

  function lockAxisWidth(w) {
    return function (axis) {
      axis.width = w;
    };
  }

  /** 左轴按 h/适水带/Hp 动态；右轴按 Hc 动态；晒田 hmin≪0 不拉垮主刻度 */
  function computeDepthAxisRange(rows) {
    var left = [];
    var right = [];
    (rows || []).forEach(function (r) {
      ["h_before", "hmax", "Hp"].forEach(function (k) {
        var v = Number(r[k]);
        if (Number.isFinite(v)) left.push(v);
      });
      var hmin = Number(r.hmin);
      if (Number.isFinite(hmin) && hmin > -40) left.push(hmin);
      var hc = Number(r.Hc);
      if (Number.isFinite(hc)) right.push(hc);
    });
    var lMin = left.length ? Math.min.apply(null, left) : 0;
    var lMax = left.length ? Math.max.apply(null, left) : 80;
    lMin = Math.min(0, Math.floor(lMin / 10) * 10);
    lMax = Math.max(60, Math.ceil(lMax / 10) * 10);
    // 主区间优先落在约 0–100；若水位更高则跟着抬
    if (lMax < 100 && lMax - lMin < 100) lMax = Math.min(100, Math.max(lMax, lMin + 60));
    var rMax = right.length ? Math.max.apply(null, right) : 100;
    rMax = Math.max(100, Math.ceil(rMax / 40) * 40);
    var lStep = lMax - lMin <= 80 ? 10 : 20;
    var rStep = rMax <= 160 ? 20 : 40;
    return { lMin: lMin, lMax: lMax, lStep: lStep, rMin: 0, rMax: rMax, rStep: rStep };
  }

  function depthChartOptions(range) {
    range = range || { lMin: 0, lMax: 100, lStep: 10, rMin: 0, rMax: 200, rStep: 40 };
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "start",
          labels: { boxWidth: 12, font: { size: 12 }, padding: 8 },
        },
      },
      scales: {
        x: timeAxisOptions(),
        y: {
          type: "linear",
          position: "left",
          min: range.lMin,
          max: range.lMax,
          title: { display: true, text: "水深 mm", font: { size: 13 } },
          ticks: { font: { size: 12 }, stepSize: range.lStep },
          grid: { color: "#e8eee9" },
          afterFit: lockAxisWidth(AXIS_LEFT_W),
        },
        y1: {
          type: "linear",
          position: "right",
          min: range.rMin,
          max: range.rMax,
          title: { display: true, text: "Hc mm", font: { size: 13 }, color: "#c62828" },
          ticks: { font: { size: 12 }, stepSize: range.rStep, color: "#c62828" },
          grid: { drawOnChartArea: false },
          afterFit: lockAxisWidth(AXIS_RIGHT_W),
        },
      },
      layout: { padding: { top: 2, bottom: 0 } },
    };
  }

  /** 灌排/降雨与水深图共用左右轴宽，保证竖线日期对齐 */
  function mmBarChartOptions(yTitle) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "start",
          labels: { boxWidth: 12, font: { size: 12 }, padding: 8 },
        },
      },
      scales: {
        x: timeAxisOptions(),
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: yTitle || "mm", font: { size: 13 } },
          ticks: { font: { size: 12 } },
          grid: { color: "#e8eee9" },
          afterFit: lockAxisWidth(AXIS_LEFT_W),
        },
        y1: {
          type: "linear",
          position: "right",
          min: 0,
          max: 1,
          title: { display: true, text: " ", font: { size: 13 } },
          ticks: { display: false },
          grid: { drawOnChartArea: false },
          afterFit: lockAxisWidth(AXIS_RIGHT_W),
        },
      },
      layout: { padding: { top: 2, bottom: 0 } },
    };
  }

  function destroyChartHolder(holder) {
    if (holder && holder.chart) {
      try {
        holder.chart.destroy();
      } catch (e) {}
      holder.chart = null;
    }
  }

  function ensureLineOrBar(holder, canvasId, type, options) {
    if (!$(canvasId) || typeof Chart === "undefined") return holder.chart;
    // 缺右轴占位时重建，保证三图 chartArea 对齐
    if (holder.chart) {
      var hasY1 = holder.chart.options && holder.chart.options.scales && holder.chart.options.scales.y1;
      if (!hasY1 && options && options.scales && options.scales.y1) {
        destroyChartHolder(holder);
      }
    }
    if (holder.chart) return holder.chart;
    holder.chart = new Chart($(canvasId), {
      type: type || "line",
      data: { labels: [], datasets: [] },
      options: options,
    });
    return holder.chart;
  }

  function applyDepthAxisRange(chart, rows) {
    if (!chart || !chart.options || !chart.options.scales) return;
    var range = computeDepthAxisRange(rows);
    chart.options.scales.y.min = range.lMin;
    chart.options.scales.y.max = range.lMax;
    chart.options.scales.y.ticks.stepSize = range.lStep;
    if (chart.options.scales.y1) {
      chart.options.scales.y1.min = range.rMin;
      chart.options.scales.y1.max = range.rMax;
      chart.options.scales.y1.ticks.stepSize = range.rStep;
    }
  }

  function buildThresholdDatasets(seriesFn, src) {
    // 适水带 + Hp 走左轴(0–100)；Hc 走右轴，避免把水深刻度拉开
    var hmin = seriesFn(src, "hmin");
    var hmax = seriesFn(src, "hmax");
    return [
      {
        type: "line",
        label: "适水带上沿 hmax",
        data: hmax,
        yAxisID: "y",
        borderColor: "rgba(45,90,66,0.55)",
        backgroundColor: "rgba(45,90,66,0.14)",
        borderWidth: 1.6,
        pointRadius: 0,
        fill: "+1",
        order: 3,
      },
      {
        type: "line",
        label: "适水带下沿 hmin",
        data: hmin,
        yAxisID: "y",
        borderColor: "rgba(154,91,18,0.8)",
        borderWidth: 1.6,
        pointRadius: 0,
        fill: false,
        order: 3,
      },
      {
        type: "line",
        label: "Hp",
        data: seriesFn(src, "Hp"),
        yAxisID: "y",
        borderColor: "#2f6fed",
        borderDash: [6, 3],
        borderWidth: 2,
        pointRadius: 0,
        order: 2,
      },
      {
        type: "line",
        label: "Hc",
        data: seriesFn(src, "Hc"),
        yAxisID: "y1",
        borderColor: "#c62828",
        borderDash: [3, 3],
        borderWidth: 2.2,
        pointRadius: 0,
        order: 2,
      },
    ];
  }

  function buildIrrigDrainBars(rows) {
    var irrig = [];
    var drain = [];
    (rows || []).forEach(function (r) {
      var a = actionMm(r);
      irrig.push(a > 0 ? a : 0);
      drain.push(a < 0 ? a : 0);
    });
    return [
      {
        type: "bar",
        label: "灌",
        data: irrig,
        backgroundColor: "rgba(26,122,69,0.55)",
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
      },
      {
        type: "bar",
        label: "排",
        data: drain,
        backgroundColor: "rgba(180,35,24,0.55)",
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
      },
    ];
  }

  function buildRainBars(rows) {
    return [
      {
        type: "bar",
        label: "降雨",
        data: (rows || []).map(rainMm),
        backgroundColor: "rgba(91,124,153,0.55)",
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
      },
    ];
  }

  function buildReviewForecastRainBars(rows) {
    return [
      {
        type: "bar",
        label: "未来7日预报累计雨量",
        data: (rows || []).map(forecastRainMm),
        backgroundColor: "rgba(91,124,153,0.55)",
        borderWidth: 0,
        barPercentage: 0.9,
        categoryPercentage: 0.95,
      },
    ];
  }

  var _seasonHolders = {
    depth: { chart: null },
    action: { chart: null },
    rain: { chart: null },
    _v: 0,
  };
  var _waterHolders = {
    depth: { chart: null },
    action: { chart: null },
    rain: { chart: null },
    _v: 0,
  };
  var CHART_LAYOUT_V = 3; // 双轴动态 + 左右轴宽锁定对齐

  function bumpChartLayout(holders) {
    if (holders._v === CHART_LAYOUT_V) return;
    destroyChartHolder(holders.depth);
    destroyChartHolder(holders.action);
    destroyChartHolder(holders.rain);
    holders._v = CHART_LAYOUT_V;
  }

  function ensureSeasonCharts() {
    bumpChartLayout(_seasonHolders);
    seasonChart = ensureLineOrBar(
      _seasonHolders.depth,
      "seasonChart",
      "line",
      depthChartOptions()
    );
    seasonActionChart = ensureLineOrBar(
      _seasonHolders.action,
      "seasonActionChart",
      "bar",
      mmBarChartOptions("灌/排 mm")
    );
    seasonRainChart = ensureLineOrBar(
      _seasonHolders.rain,
      "seasonRainChart",
      "bar",
      mmBarChartOptions("降雨 mm")
    );
  }

  function ensureWaterCharts() {
    bumpChartLayout(_waterHolders);
    waterChart = ensureLineOrBar(
      _waterHolders.depth,
      "waterChart",
      "line",
      depthChartOptions()
    );
    waterActionChart = ensureLineOrBar(
      _waterHolders.action,
      "waterActionChart",
      "bar",
      mmBarChartOptions("灌/排 mm")
    );
    waterRainChart = ensureLineOrBar(
      _waterHolders.rain,
      "waterRainChart",
      "bar",
      mmBarChartOptions("降雨 mm")
    );
  }

  async function updateSeasonChart(year) {
    var wrap = $("seasonChartWrap");
    var yWrap = $("yearlyChartWrap");
    if (!year) {
      setHidden(wrap, true);
      setHidden(yWrap, false);
      return;
    }
    setHidden(yWrap, true);
    setHidden(wrap, false);
    if ($("seasonCaption"))
      $("seasonCaption").textContent = year + " 整季水层 · 左轴水深 · 右轴 Hc";
    try {
      var j = await fetch("/api/run/season?year=" + encodeURIComponent(year)).then(function (r) {
        return r.json();
      });
      if (!j.ok) return;
      ensureSeasonCharts();
      if (!seasonChart) return;
      var ppo = j.ppo || [];
      var rule = j.rule || [];
      var labels = ppo.length
        ? ppo.map(function (r) {
            return r.date;
          })
        : rule.map(function (r) {
            return r.date;
          });
      function series(rows, key) {
        var m = {};
        (rows || []).forEach(function (r) {
          m[r.date] = r[key];
        });
        return labels.map(function (d) {
          return m[d] != null ? m[d] : null;
        });
      }
      var thrSrc = ppo.length ? ppo : rule;
      var dateLabels = labels.map(formatDateOnly);
      var actRows = ppo.length ? ppo : rule;

      applyDepthAxisRange(seasonChart, thrSrc);
      // 评估单年水层只画 PPO（基线对比留给上方 KPI / 分年柱图；审阅台另有整季图）
      seasonChart.data.labels = dateLabels;
      seasonChart.data.datasets = [
        {
          type: "line",
          label: "PPO 水深 h",
          data: series(ppo, "h_before"),
          yAxisID: "y",
          borderColor: "#1a4d32",
          borderWidth: 2.6,
          pointRadius: 0,
          tension: 0.12,
          order: 0,
        },
      ].concat(buildThresholdDatasets(series, thrSrc));
      seasonChart.update("none");
      if ($("seasonCaption"))
        $("seasonCaption").textContent = year + " 整季水层 · 左轴水深(动态) · 右轴 Hc";

      if (seasonActionChart) {
        seasonActionChart.data.labels = dateLabels;
        seasonActionChart.data.datasets = buildIrrigDrainBars(actRows);
        seasonActionChart.update("none");
      }
      if (seasonRainChart) {
        seasonRainChart.data.labels = dateLabels;
        seasonRainChart.data.datasets = buildRainBars(actRows);
        seasonRainChart.update("none");
      }
    } catch (e) {}
  }

  function renderEvalView() {
    var body = $("summaryBody");
    if (!body || !lastSummary) return;
    var year = $("evalYearFilter") ? $("evalYearFilter").value : "";
    var bits = [];
    var rep = lastSummary.report && lastSummary.report.training_summary;
    if (rep && !year) {
      bits.push(
        '<div class="metric-cards">' +
          "<div><b>" +
          esc(rep.total_episodes) +
          "</b><span>训练局数</span></div>" +
          "<div><b>" +
          fmt(rep.final_mean_return, 1) +
          "</b><span>末段均值回报</span></div>" +
          "<div><b>" +
          fmt(rep.total_time_hours, 2) +
          "</b><span>耗时 h</span></div>" +
          "<div><b>" +
          fmt(rep.final_mean_actions, 1) +
          "</b><span>末段均动作</span></div>" +
          "</div>"
      );
    }
    if ($("evalViewHint")) {
      $("evalViewHint").textContent = year
        ? "当前：" + year + " 年 KPI + PPO 整季水深/灌排/实测雨（审阅完成后）"
        : "当前：多年均值 KPI + 下方分年柱图（雨利用率→灌→排→管理水量→产量系数）";
    }
    if (!year) {
      if (lastSummary.compare_summary) {
        bits.push("<h4 style='margin:0.6rem 0 0.35rem'>评估集 KPI（多年均值）</h4>");
        bits.push(renderRunCompareTable(lastSummary.compare_summary));
      }
      setHidden($("yearlyChartWrap"), false);
      setHidden($("seasonChartWrap"), true);
      updateYearlyChart(lastSummary.yearly_ppo, lastSummary.yearly_rule);
    } else {
      var pair = yearlyPairToCompare(
        findYearlyRow(lastSummary.yearly_ppo, year),
        findYearlyRow(lastSummary.yearly_rule, year)
      );
      bits.push("<h4 style='margin:0.6rem 0 0.35rem'>评估集 KPI（" + esc(year) + " 整季）</h4>");
      if (pair) bits.push(renderRunCompareTable(pair, { title: year + " 年 KPI" }));
      else bits.push('<p class="hint">该年无 yearly 指标。</p>');
      updateSeasonChart(year);
    }
    if (lastSummary.has_best) bits.push('<p class="ok-line">已有 best 权重</p>');
    if (!bits.length) bits.push('<p class="hint">暂无摘要。</p>');
    body.innerHTML = bits.join("");
  }

  function renderSummary(summary) {
    var body = $("summaryBody");
    if (!body) return;
    lastSummary = summary || null;
    if (!summary) {
      body.innerHTML = '<p class="hint">暂无摘要（完成训评或加载历史训练会话）。</p>';
      return;
    }
    lastRunCompare = summary.compare_summary || null;
    lastYearly = { ppo: summary.yearly_ppo || [], rule: summary.yearly_rule || [] };
    var years = (summary.yearly_ppo || []).map(function (r) {
      return String(r.year);
    });
    fillEvalYearSelect(years);
    var conv = summary.report && summary.report.convergence_analysis;
    if ($("convHint")) {
      if (conv) {
        $("convHint").textContent =
          "收敛：" +
          (conv.convergence_status || "—") +
          " · 回报改善 " +
          fmt(conv.return_improvement, 2) +
          " · 动作改善 " +
          fmt(conv.action_improvement, 2) +
          " · 稳定度 " +
          fmt(conv.return_stability, 3);
      } else {
        $("convHint").textContent = summary.train_loss
          ? "已有 training_loss（新训可见损失曲线）。"
          : "本训练会话无 training_loss.csv（旧跑次常见；重新训练后可看损失曲线）。";
      }
    }
    renderEvalView();
    renderComparePanel();
  }

  /** 从 task_compare / three_way 抽三方表 */
  function pickArm(obj, keys) {
    if (!obj) return null;
    for (var i = 0; i < keys.length; i++) {
      if (obj[keys[i]] != null) return obj[keys[i]];
    }
    return null;
  }

  function hasKpiFields(obj) {
    if (!obj || typeof obj !== "object") return false;
    return (
      obj.irrigation_total != null ||
      obj.rain_utilization != null ||
      obj.managed_water != null ||
      obj.drainage_total != null ||
      obj.yield_factor != null
    );
  }

  function armKpiBag(arm, prefix) {
    if (!arm || typeof arm !== "object") return null;
    if (arm.means && hasKpiFields(arm.means)) return arm.means;
    var s = arm.summary;
    if (s && typeof s === "object") {
      var out = {};
      var keys = [
        "irrigation_total",
        "drainage_total",
        "managed_water",
        "rain_utilization",
        "yield_factor",
        "irrigation_times",
        "drainage_times",
      ];
      var pref = prefix || "ppo_";
      keys.forEach(function (k) {
        if (s[pref + k] != null) out[k] = Number(s[pref + k]);
        else if (s[k] != null) out[k] = Number(s[k]);
      });
      return Object.keys(out).length ? out : null;
    }
    // 已是扁平 KPI 行（yearly 单年 / means）
    if (
      arm.irrigation_total != null ||
      arm.rain_utilization != null ||
      arm.managed_water != null
    ) {
      return arm;
    }
    return null;
  }

  function destroyCompareCharts() {
    (compareCharts || []).forEach(function (ch) {
      try {
        if (ch && ch.destroy) ch.destroy();
      } catch (e) {}
    });
    compareCharts = [];
  }

  function destroyChartOnCanvas(canvasOrId) {
    var el =
      typeof canvasOrId === "string" ? $(canvasOrId) : canvasOrId;
    if (!el || typeof Chart === "undefined") return;
    try {
      var existing = Chart.getChart ? Chart.getChart(el) : null;
      if (existing) {
        compareCharts = (compareCharts || []).filter(function (c) {
          return c !== existing;
        });
        existing.destroy();
      }
    } catch (e) {}
  }

  function trackCompareChart(ch) {
    if (ch) compareCharts.push(ch);
    return ch;
  }

  function toYmd(dateStr) {
    var s = String(dateStr || "")
      .trim()
      .replace(/\//g, "-");
    var m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
    if (!m) return s;
    var mm = m[2].length === 1 ? "0" + m[2] : m[2];
    var dd = m[3].length === 1 ? "0" + m[3] : m[3];
    return m[1] + "-" + mm + "-" + dd;
  }

  function renderThreeWayTable(tc, meansApi) {
    if (!tc || typeof tc !== "object") return "";
    meansApi = meansApi || {};
    var arms = tc.arms && typeof tc.arms === "object" ? tc.arms : tc;
    var rule =
      (hasKpiFields(meansApi.rule) ? meansApi.rule : null) ||
      armKpiBag(pickArm(arms, ["rule", "RULE", "baseline"]), "rule_") ||
      armKpiBag(tc, "rule_");
    var ppo =
      (hasKpiFields(meansApi.original_ppo) ? meansApi.original_ppo : null) ||
      armKpiBag(pickArm(arms, ["original_ppo", "ppo", "PPO", "original"]), "ppo_");
    var hitl =
      (hasKpiFields(meansApi.hitl_ppo) ? meansApi.hitl_ppo : null) ||
      armKpiBag(pickArm(arms, ["hitl_ppo", "hitl", "HITL"]), "ppo_");
    var metrics = [
      { key: "irrigation_total", label: "灌总量 mm", lowerBetter: true },
      { key: "drainage_total", label: "排总量 mm", lowerBetter: true },
      { key: "managed_water", label: "管理水量 mm", lowerBetter: true },
      { key: "rain_utilization", label: "雨利用率", lowerBetter: false },
      { key: "yield_factor", label: "产量水分系数", lowerBetter: false },
      { key: "irrigation_times", label: "灌水日数", lowerBetter: true },
      { key: "drainage_times", label: "排水日数", lowerBetter: true },
      { key: "irrigation_contiguous_segments", label: "连续灌水段数", lowerBetter: true },
      { key: "drainage_contiguous_segments", label: "连续排水段数", lowerBetter: true },
      { key: "stress_days", label: "胁迫日数(ks)", lowerBetter: true },
      {
        key: "irrigation_small_run_ge3d_segments",
        label: "小额灌≥3日段数",
        lowerBetter: true,
        alwaysShow: true,
      },
      {
        key: "irrigation_small_run_ge3d_days",
        label: "小额灌≥3日天数",
        lowerBetter: true,
        alwaysShow: true,
      },
      {
        key: "irrigation_small_run_ge3d_mm",
        label: "小额灌≥3日水量",
        lowerBetter: true,
        alwaysShow: true,
      },
    ];

    function val(arm, key) {
      if (!arm) return null;
      if (arm[key] != null && arm[key] !== "") return Number(arm[key]);
      var pref = ["mean_", "avg_", ""];
      for (var i = 0; i < pref.length; i++) {
        var k = pref[i] + key;
        if (arm[k] != null && arm[k] !== "") return Number(arm[k]);
      }
      return null;
    }

    function delta(a, b, lowerBetter) {
      if (a == null || b == null || Math.abs(b) < 1e-12) return null;
      var pct = ((a - b) / Math.abs(b)) * 100;
      return lowerBetter ? -pct : pct;
    }

    if (!ppo && !hitl && (tc.ppo_irrigation_total != null || tc.rule_irrigation_total != null)) {
      return renderRunCompareTable(tc);
    }

    var body = metrics
      .map(function (m) {
        var rv = val(rule, m.key);
        var pv = val(ppo, m.key);
        var hv = val(hitl, m.key);
        if (rv == null && tc["rule_" + m.key] != null) rv = Number(tc["rule_" + m.key]);
        if (pv == null && tc["ppo_" + m.key] != null) pv = Number(tc["ppo_" + m.key]);
        if (hv == null && tc["hitl_" + m.key] != null) hv = Number(tc["hitl_" + m.key]);
        if (m.alwaysShow) {
          if (rv == null || !Number.isFinite(rv)) rv = 0;
          if (pv == null || !Number.isFinite(pv)) pv = 0;
          if (hv == null || !Number.isFinite(hv)) hv = 0;
        }
        if (rv == null && pv == null && hv == null) return "";
        var dPpo = delta(pv, rv, m.lowerBetter);
        var dHitl = delta(hv, rv, m.lowerBetter);
        return (
          "<tr><td>" +
          esc(m.label) +
          "</td><td>" +
          fmt(rv, 3) +
          "</td><td>" +
          fmt(pv, 3) +
          '</td><td class="' +
          deltaClass(dPpo) +
          '">' +
          fmtPct(dPpo) +
          "</td><td>" +
          fmt(hv, 3) +
          '</td><td class="' +
          deltaClass(dHitl) +
          '">' +
          fmtPct(dHitl) +
          "</td></tr>"
        );
      })
      .join("");

    return (
      '<p class="hint" style="margin-top:0">Δ% = 相对固定规则基线（绿=更好）。灌/排日数≠连续操作段数。小额灌判据：连续≥3日且每日执行量 0&lt;m≤20 mm（本轮可为 0）。胁迫日数按环境 ks，≠低于 hmin 日数。</p>' +
      '<table class="compare-table"><thead><tr>' +
      "<th>指标</th><th>基线(规则)</th><th>原 PPO</th><th>Δ% vs 基线</th><th>HITL</th><th>Δ% vs 基线</th>" +
      "</tr></thead><tbody>" +
      body +
      "</tbody></table>"
    );
  }

  function yearRoleShort(y, rolesMap) {
    var r = (rolesMap && rolesMap[String(y)]) || {};
    if (r.short) return r.short;
    var yi = parseInt(y, 10);
    if (yi >= 2017) return "留出";
    if (String(y) === "2016") return "审阅·验证";
    if (yi >= 2012 && yi <= 2014) return "审阅·训练";
    return "";
  }

  function fillCompareYearSelect(years, rolesMap) {
    var sel = $("compareYearFilter");
    if (!sel) return;
    var cur = sel.value;
    sel.innerHTML =
      '<option value="">多年均值（默认）· 含全部评估年</option>' +
      (years || [])
        .map(function (y) {
          var tag = yearRoleShort(y, rolesMap);
          return (
            '<option value="' +
            esc(y) +
            '">' +
            esc(y) +
            (tag ? " · " + tag : "") +
            "</option>"
          );
        })
        .join("");
    if (cur && years && years.indexOf(String(cur)) >= 0) sel.value = cur;
  }

  function renderThreeWayYearRow(yearRow, year) {
    if (!yearRow) return '<p class="hint">该年无三方 yearly 数据，请先跑三方对比。</p>';
    var fake = {
      rule: yearRow.rule,
      original_ppo: yearRow.original_ppo,
      hitl_ppo: yearRow.hitl_ppo,
    };
    return (
      "<h3 style='margin-top:0'>三方对比 · " +
      esc(year) +
      " 整季</h3>" +
      renderThreeWayTable(fake)
    );
  }

  function compareVizShellHtml(years, pairsPayload) {
    var rolesMap = {};
    (pairsPayload && pairsPayload.pairs ? pairsPayload.pairs : []).forEach(function (p) {
      rolesMap[String(p.year)] = p;
    });
    var yopts = (years || [])
      .map(function (y) {
        var sel = String(y) === "2016" ? " selected" : "";
        var tag = yearRoleShort(y, rolesMap);
        return (
          '<option value="' +
          esc(y) +
          '"' +
          sel +
          ">" +
          esc(y) +
          (tag ? " · " + tag : "") +
          "</option>"
        );
      })
      .join("");
    var guide =
      '<div class="compare-role-band">' +
      "<span><strong style=\"color:#2f6fed\">2012—2014</strong> 审阅年 · BC训练标签年</span>" +
      "<span><strong style=\"color:#c45c26\">2016</strong> 审阅年 · BC内部验证年</span>" +
      "<span><strong style=\"color:#6b7280\">2017—2020</strong> 未审阅留出年</span>" +
      "</div>" +
      '<p class="hint" style="margin:0 0 0.45rem">审阅年同样参与评估：与原PPO在相同年份、相同气象情景下对比，不是「只训练不评估」。</p>';
    return (
      guide +
      '<div class="compare-viz-block" id="comparePairBlock">' +
      "<h3>① 全年份对比 · 规则 / 原 PPO / HITL</h3>" +
      '<p class="hint" style="margin:0 0 0.4rem">每年三根柱：蓝灰=规则、绿=原PPO、橙=HITL。横轴直接标年份；背景色带区分审阅·训练 / 审阅·验证 / 留出。悬停看数值与 HITL−原PPO 差值。</p>' +
      '<div class="compare-legend">' +
      '<span><i style="background:#2f6fed"></i>审阅·训练 2012–14</span>' +
      '<span><i style="background:#c45c26"></i>审阅·验证 2016</span>' +
      '<span><i style="background:#6b7280"></i>留出 2017–20</span>' +
      '<span style="color:#5b7c99">■ 规则</span>' +
      '<span style="color:#1a7a45">■ 原PPO</span>' +
      '<span style="color:#c45c26">■ HITL</span>' +
      "</div>" +
      '<div class="compare-pair-grid" id="comparePairGrid"></div>' +
      "</div>" +
      '<div class="compare-viz-block" id="compareHeatBlock">' +
      "<h3>② 生育期 × 年份 变化热图（HITL − 原PPO）</h3>" +
      '<div class="compare-viz-toolbar">' +
      "<label>指标<select id=\"compareHeatMetric\">" +
      '<option value="irrigation_total">灌水量</option>' +
      '<option value="drainage_total">排水量</option>' +
      '<option value="operation_days">操作日数</option>' +
      '<option value="irrigation_times">灌水日数</option>' +
      '<option value="drainage_times">排水日数</option>' +
      '<option value="h_before_mean">晨水层均值</option>' +
      '<option value="stress_days">胁迫日数</option>' +
      "</select></label>" +
      '<span class="hint">颜色以「无变化」为中心：暖=HITL更大，冷=HITL更小</span>' +
      "</div>" +
      '<div class="compare-heatmap-wrap" id="compareHeatTable"></div>' +
      "</div>" +
      '<div class="compare-viz-block" id="compareProcessBlock">' +
      "<h3>③ 单年生育季过程 · 水层与执行时机</h3>" +
      '<div class="compare-viz-toolbar">' +
      "<label>年份<select id=\"compareProcessYear\">" +
      yopts +
      "</select></label>" +
      '<span class="hint" id="compareProcessRole"></span>' +
      "</div>" +
      '<p class="hint" style="margin:0 0 0.35rem">上：实测降雨；中：规则/原PPO/HITL 水层 + 阈值；下：执行量（灌+/排−）。审阅年红色竖线=专家修改日。</p>' +
      '<div class="compare-process-stack">' +
      '<div class="chart-box"><canvas id="cmpRainChart"></canvas></div>' +
      '<div class="chart-box chart-box-tall"><canvas id="cmpDepthChart"></canvas></div>' +
      '<div class="chart-box"><canvas id="cmpExecChart"></canvas></div>' +
      "</div>" +
      '<div id="compareProcessNote" class="hint"></div>' +
      "</div>" +
      '<div class="compare-viz-block" id="compareStageBlock">' +
      "<h3>④ 按生育期表 · PPO vs HITL</h3>" +
      '<div class="compare-viz-toolbar">' +
      "<label>范围<select id=\"compareStageScope\">" +
      '<option value="year:2016">2016 · 审阅·验证</option>' +
      '<option value="holdout">留出年 2017–2020 均值</option>' +
      (years || [])
        .map(function (y) {
          if (String(y) === "2016") return "";
          return (
            '<option value="year:' +
            esc(y) +
            '">' +
            esc(y) +
            " · " +
            yearRoleShort(y, rolesMap) +
            "</option>"
          );
        })
        .join("") +
      "</select></label>" +
      "</div>" +
      '<div id="compareStageTable"></div>' +
      "</div>" +
      '<div class="compare-viz-block" id="compareAbsorbBlock">' +
      "<h3>⑤ 专家动作吸收 · 概率变化</h3>" +
      '<p class="hint" style="margin:0 0 0.35rem" id="compareAbsorbSummary">加载中…</p>' +
      '<div class="compare-absorb-grid">' +
      '<div class="compare-pair-card">' +
      '<div class="pair-title">改动日：专家动作概率（原PPO→HITL）</div>' +
      '<p class="hint" style="margin:0.15rem 0 0.35rem">每行一个日期（YYYY-MM-DD）；绿=原PPO，橙=HITL，右侧为 Δ</p>' +
      '<div class="absorb-edit-legend"><span class="ae-ppo">原PPO</span><span class="ae-hitl">HITL</span></div>' +
      '<div class="absorb-edit-scroll" id="cmpAbsorbEditList"></div>' +
      "</div>" +
      '<div class="compare-pair-card">' +
      '<div class="pair-title">确认日：原动作概率密度（分箱，避免 n 过大叠点）</div>' +
      '<p class="hint" id="compareAbsorbKeepHint" style="margin:0.15rem 0 0.35rem"></p>' +
      '<div class="chart-box chart-box-absorb"><canvas id="cmpAbsorbKeepChart"></canvas></div>' +
      '<div class="pair-title" style="margin-top:0.45rem;font-size:0.92rem">确认日：概率变化 Δ=HITL−原PPO</div>' +
      '<div class="chart-box chart-box-absorb-delta"><canvas id="cmpAbsorbKeepDeltaChart"></canvas></div>' +
      "</div>" +
      "</div>" +
      "</div>"
    );
  }

  function mountPairCharts(pairsPayload) {
    var grid = $("comparePairGrid");
    if (!grid || !pairsPayload || !pairsPayload.pairs) return;
    var pairs = pairsPayload.pairs || [];
    var years = pairs.map(function (p) {
      return String(p.year);
    });
    var yearLabels = pairs.map(function (p) {
      return String(p.year);
    });
    var metrics = [
      { key: "irrigation_total", label: "灌水量 mm", digits: 0 },
      { key: "drainage_total", label: "排水量 mm", digits: 0 },
      { key: "irrigation_times", label: "灌水日数", digits: 0 },
      { key: "drainage_times", label: "排水日数", digits: 0 },
      { key: "stress_days", label: "胁迫日数", digits: 0 },
      { key: "managed_water", label: "管理水量 mm", digits: 0 },
    ];

    function armVal(p, arm, key) {
      var row = p && p[arm];
      if (!row || row[key] == null || row[key] === "") return null;
      var n = Number(row[key]);
      return Number.isFinite(n) ? n : null;
    }

    grid.innerHTML = metrics
      .map(function (m, idx) {
        var deltaBits = pairs
          .map(function (p) {
            var pv = armVal(p, "original_ppo", m.key);
            var hv = armVal(p, "hitl_ppo", m.key);
            var d = pv != null && hv != null ? hv - pv : null;
            var cls =
              d == null ? "" : d < -1e-9 ? "down" : d > 1e-9 ? "up" : "flat";
            var txt =
              d == null
                ? "—"
                : (d > 0 ? "+" : "") + Number(d).toFixed(m.digits);
            return (
              '<span class="pair-delta-chip ' +
              cls +
              '" title="HITL−原PPO">' +
              esc(p.year) +
              " " +
              txt +
              "</span>"
            );
          })
          .join("");
        return (
          '<div class="compare-pair-card">' +
          '<div class="pair-title-row"><div class="pair-title">' +
          esc(m.label) +
          '</div><div class="pair-delta-row" title="各年 HITL 相对原PPO">' +
          deltaBits +
          "</div></div>" +
          '<div class="chart-box chart-box-pair"><canvas id="cmpPair' +
          idx +
          '"></canvas></div></div>'
        );
      })
      .join("");

    var yearBandPlugin = {
      id: "yearRoleBands",
      beforeDraw: function (chart) {
        var x = chart.scales.x;
        var y = chart.scales.y;
        if (!x || !y) return;
        var ctx = chart.ctx;
        pairs.forEach(function (p, i) {
          var col =
            p.role === "bc_train"
              ? "rgba(47,111,237,0.10)"
              : p.role === "bc_val"
                ? "rgba(196,92,38,0.12)"
                : p.role === "holdout"
                  ? "rgba(107,114,128,0.08)"
                  : "transparent";
          var x0 = x.getPixelForValue(i) - (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
          var x1 = x.getPixelForValue(i) + (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
          if (years.length === 1) {
            x0 = x.left;
            x1 = x.right;
          } else if (i === 0) {
            x0 = x.left;
          } else if (i === years.length - 1) {
            x1 = x.right;
          }
          ctx.save();
          ctx.fillStyle = col;
          ctx.fillRect(x0, y.top, Math.max(0, x1 - x0), y.bottom - y.top);
          ctx.restore();
        });
      },
    };

    metrics.forEach(function (m, idx) {
      var canvas = $("cmpPair" + idx);
      if (!canvas || typeof Chart === "undefined") return;
      var ruleYs = pairs.map(function (p) {
        return armVal(p, "rule", m.key);
      });
      var ppoYs = pairs.map(function (p) {
        return armVal(p, "original_ppo", m.key);
      });
      var hitlYs = pairs.map(function (p) {
        return armVal(p, "hitl_ppo", m.key);
      });
      trackCompareChart(
        new Chart(canvas, {
          type: "bar",
          data: {
            labels: yearLabels,
            datasets: [
              {
                label: "规则",
                data: ruleYs,
                backgroundColor: "rgba(91,124,153,0.75)",
                borderColor: ARM_COLORS.rule,
                borderWidth: 1,
                borderRadius: 2,
              },
              {
                label: "原PPO",
                data: ppoYs,
                backgroundColor: "rgba(26,122,69,0.80)",
                borderColor: ARM_COLORS.original_ppo,
                borderWidth: 1,
                borderRadius: 2,
              },
              {
                label: "HITL",
                data: hitlYs,
                backgroundColor: "rgba(196,92,38,0.85)",
                borderColor: ARM_COLORS.hitl_ppo,
                borderWidth: 1,
                borderRadius: 2,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
              legend: {
                position: "top",
                align: "end",
                labels: { boxWidth: 12, font: { size: 11 }, padding: 8 },
              },
              tooltip: {
                callbacks: {
                  title: function (items) {
                    if (!items.length) return "";
                    var i = items[0].dataIndex;
                    var p = pairs[i] || {};
                    return (
                      yearLabels[i] +
                      (p.short ? " · " + p.short : "") +
                      (p.label ? "（" + p.label + "）" : "")
                    );
                  },
                  afterBody: function (items) {
                    if (!items.length) return [];
                    var i = items[0].dataIndex;
                    var pv = ppoYs[i];
                    var hv = hitlYs[i];
                    if (pv == null || hv == null) return [];
                    var d = hv - pv;
                    return [
                      "HITL − 原PPO = " +
                        (d > 0 ? "+" : "") +
                        d.toFixed(m.digits),
                    ];
                  },
                },
              },
            },
            scales: {
              x: {
                stacked: false,
                ticks: {
                  font: { size: 12, weight: "600" },
                  maxRotation: 0,
                  autoSkip: false,
                },
                grid: { display: false },
              },
              y: {
                beginAtZero: true,
                ticks: { font: { size: 11 } },
                grid: { color: "#eef2ef" },
                title: { display: false },
              },
            },
            datasets: {
              bar: {
                categoryPercentage: 0.72,
                barPercentage: 0.9,
              },
            },
          },
          plugins: [yearBandPlugin],
        })
      );
    });
  }

  function divergingColor(delta, absMax) {
    if (delta == null || !Number.isFinite(delta)) return "#f0f2f0";
    var t = Math.max(-1, Math.min(1, delta / (absMax || 1)));
    if (Math.abs(t) < 0.02) return "#f7f7f7";
    if (t > 0) {
      var a = Math.min(0.85, 0.15 + Math.abs(t) * 0.7);
      return "rgba(196,92,38," + a.toFixed(2) + ")";
    }
    var b = Math.min(0.85, 0.15 + Math.abs(t) * 0.7);
    return "rgba(47,111,237," + b.toFixed(2) + ")";
  }

  async function mountHeatmap() {
    var box = $("compareHeatTable");
    var sel = $("compareHeatMetric");
    if (!box) return;
    var metric = sel ? sel.value : "irrigation_total";
    try {
      var j = await compareFetchJson(
        "/api/compare/heatmap?metric=" + encodeURIComponent(metric)
      );
      if (!j.ok) {
        box.innerHTML = '<p class="hint">' + esc(j.error || "热图不可用") + "</p>";
        return;
      }
      var absMax = Math.max(
        Math.abs(j.scale && j.scale.max != null ? j.scale.max : 1),
        Math.abs(j.scale && j.scale.min != null ? j.scale.min : 1),
        1e-6
      );
      var head =
        "<tr><th>生育期 \\ 年</th>" +
        (j.rows || [])
          .map(function (r) {
            return (
              "<th>" +
              esc(r.year) +
              '<span class="yr-role">' +
              esc(r.short || "") +
              "</span></th>"
            );
          })
          .join("") +
        "</tr>";
      var stages = j.stages || [];
      var body = stages
        .map(function (st, si) {
          var cells = (j.rows || [])
            .map(function (r) {
              var c = (r.cells && r.cells[si]) || {};
              var d = c.delta;
              var title =
                (c.stage_cn || "") +
                " " +
                r.year +
                ": PPO " +
                fmt(c.ppo, 1) +
                " → HITL " +
                fmt(c.hitl, 1) +
                " (Δ " +
                fmt(d, 1) +
                ")";
              return (
                '<td class="hm-cell" style="background:' +
                divergingColor(d, absMax) +
                '" title="' +
                esc(title) +
                '">' +
                (d == null ? "—" : fmt(d, 1)) +
                "</td>"
              );
            })
            .join("");
          return (
            "<tr><td>" + esc(st.stage_cn || st.stage) + "</td>" + cells + "</tr>"
          );
        })
        .join("");
      box.innerHTML =
        '<p class="hint" style="margin:0.4rem 0.5rem">' +
        esc(j.metric_label || "") +
        " · " +
        esc(j.note || "") +
        "</p>" +
        '<table class="compare-heatmap"><thead>' +
        head +
        "</thead><tbody>" +
        body +
        "</tbody></table>";
    } catch (e) {
      box.innerHTML = '<p class="hint">' + esc(String(e.message || e)) + "</p>";
    }
  }

  function stageBoundaryPlugin(rows) {
    return {
      id: "stageBounds",
      afterDraw: function (chart) {
        if (!rows || !rows.length) return;
        var xAxis = chart.scales.x;
        var yAxis = chart.scales.y;
        if (!xAxis || !yAxis) return;
        var ctx = chart.ctx;
        var last = null;
        var lastLabelX = -999;
        rows.forEach(function (r, i) {
          var st = r.stage_cn || r.stage || "";
          if (st && st !== last) {
            var x = xAxis.getPixelForValue(i);
            ctx.save();
            ctx.strokeStyle = "rgba(0,0,0,0.16)";
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(x, yAxis.top);
            ctx.lineTo(x, yAxis.bottom);
            ctx.stroke();
            // 标签过密则跳过文字，只保留竖线
            if (x - lastLabelX >= 36) {
              ctx.fillStyle = "rgba(0,0,0,0.55)";
              ctx.font = "10px sans-serif";
              var lab = String(st);
              if (lab.length > 5) lab = lab.slice(0, 5);
              ctx.fillText(lab, x + 3, yAxis.top + 11);
              lastLabelX = x;
            }
            ctx.restore();
            last = st;
          }
        });
      },
    };
  }

  function expertEditPlugin(baseRows, editDays) {
    var editSet = {};
    (editDays || []).forEach(function (e) {
      editSet[String(e.date || "").replace(/-/g, "/")] = e;
    });
    var indices = [];
    (baseRows || []).forEach(function (r, i) {
      var d = String(r.date || "").replace(/-/g, "/");
      if (editSet[d]) indices.push(i);
    });
    return {
      id: "expertEdits",
      afterDraw: function (chart) {
        if (!indices.length) return;
        var xAxis = chart.scales.x;
        var yAxis = chart.scales.y;
        if (!xAxis || !yAxis) return;
        var ctx = chart.ctx;
        indices.forEach(function (i) {
          var x = xAxis.getPixelForValue(i);
          ctx.save();
          ctx.strokeStyle = "rgba(180,35,24,0.75)";
          ctx.lineWidth = 2;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(x, yAxis.top);
          ctx.lineTo(x, yAxis.bottom);
          ctx.stroke();
          ctx.fillStyle = "rgba(180,35,24,0.9)";
          ctx.beginPath();
          ctx.moveTo(x, yAxis.top);
          ctx.lineTo(x - 3, yAxis.top + 6);
          ctx.lineTo(x + 3, yAxis.top + 6);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        });
      },
    };
  }

  async function mountProcessCharts(year) {
    year = year || "2016";
    try {
      var j = await compareFetchJson(
        "/api/compare/season?year=" + encodeURIComponent(year)
      );
      if (!j.ok) {
        if ($("compareProcessNote"))
          $("compareProcessNote").textContent = j.error || "无法加载整季轨迹";
        return;
      }
      if ($("compareProcessRole") && j.year_role)
        $("compareProcessRole").textContent = j.year_role.label || "";
      var rule = (j.arms && j.arms.rule) || [];
      var ppo = (j.arms && j.arms.original_ppo) || [];
      var hitl = (j.arms && j.arms.hitl_ppo) || [];
      var base = hitl.length ? hitl : ppo.length ? ppo : rule;
      var editDays = j.expert_edit_days || [];
      var labels = base.map(function (r) {
        return String(r.date || "").slice(5);
      });
      var rainCanvas = $("cmpRainChart");
      var depthCanvas = $("cmpDepthChart");
      var execCanvas = $("cmpExecChart");
      var editPlug = expertEditPlugin(base, editDays);
      if (rainCanvas && typeof Chart !== "undefined") {
        trackCompareChart(
          new Chart(rainCanvas, {
            type: "bar",
            data: {
              labels: labels,
              datasets: [
                {
                  label: "实测雨 mm",
                  data: base.map(function (r) {
                    return r.rain != null ? r.rain : r.P_1_a;
                  }),
                  backgroundColor: "rgba(91,124,153,0.55)",
                  borderWidth: 0,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: true } },
              scales: {
                x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
                y: { title: { display: true, text: "雨 mm" } },
              },
            },
            plugins: [editPlug],
          })
        );
      }
      if (depthCanvas && typeof Chart !== "undefined") {
        var thr = base;
        var depthRange = computeDepthAxisRange(
          [].concat(rule, ppo, hitl, thr)
        );
        // 晒田/落干 hmin≪0：图上断开，避免左轴被拉到 -200
        var hminPlot = thr.map(function (r) {
          var v = Number(r.hmin);
          return Number.isFinite(v) && v > -40 ? v : null;
        });
        var depthCh = new Chart(depthCanvas, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "规则 h",
                data: rule.map(function (r) {
                  return r.h_before;
                }),
                yAxisID: "y",
                borderColor: ARM_COLORS.rule,
                borderDash: [4, 3],
                pointRadius: 0,
                borderWidth: 1.5,
                tension: 0.15,
              },
              {
                label: "原PPO h",
                data: ppo.map(function (r) {
                  return r.h_before;
                }),
                yAxisID: "y",
                borderColor: ARM_COLORS.original_ppo,
                pointRadius: 0,
                borderWidth: 2,
                tension: 0.15,
              },
              {
                label: "HITL h",
                data: hitl.map(function (r) {
                  return r.h_before;
                }),
                yAxisID: "y",
                borderColor: ARM_COLORS.hitl_ppo,
                pointRadius: 0,
                borderWidth: 2.2,
                tension: 0.15,
              },
              {
                label: "hmin",
                data: hminPlot,
                yAxisID: "y",
                borderColor: "rgba(0,0,0,0.35)",
                borderDash: [2, 2],
                pointRadius: 0,
                borderWidth: 1,
                spanGaps: false,
              },
              {
                label: "hmax",
                data: thr.map(function (r) {
                  return r.hmax;
                }),
                yAxisID: "y",
                borderColor: "rgba(0,0,0,0.25)",
                borderDash: [2, 2],
                pointRadius: 0,
                borderWidth: 1,
              },
              {
                label: "Hp",
                data: thr.map(function (r) {
                  return r.Hp;
                }),
                yAxisID: "y",
                borderColor: "#8d6e63",
                pointRadius: 0,
                borderWidth: 1,
              },
              {
                label: "Hc",
                data: thr.map(function (r) {
                  return r.Hc;
                }),
                borderColor: "#c62828",
                pointRadius: 0,
                borderWidth: 1.2,
                yAxisID: "y1",
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
              legend: {
                display: true,
                labels: { boxWidth: 12, font: { size: 10 } },
              },
            },
            scales: {
              x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
              y: {
                type: "linear",
                position: "left",
                min: depthRange.lMin,
                max: depthRange.lMax,
                title: { display: true, text: "水层 mm" },
                ticks: { stepSize: depthRange.lStep, font: { size: 10 } },
                grid: { color: "#e8eee9" },
              },
              y1: {
                type: "linear",
                position: "right",
                min: depthRange.rMin,
                max: depthRange.rMax,
                grid: { drawOnChartArea: false },
                title: {
                  display: true,
                  text: "Hc mm",
                  color: "#c62828",
                },
                ticks: {
                  stepSize: depthRange.rStep,
                  color: "#c62828",
                  font: { size: 10 },
                },
              },
            },
          },
          plugins: [stageBoundaryPlugin(base), editPlug],
        });
        trackCompareChart(depthCh);
      }
      if (execCanvas && typeof Chart !== "undefined") {
        function execSeries(rows, label, color, dash) {
          return {
            label: label,
            data: (rows || []).map(function (r) {
              return r.exec_signed != null
                ? r.exec_signed
                : (r.m || 0) - (r.d || 0);
            }),
            borderColor: color,
            backgroundColor: color,
            borderDash: dash || [],
            pointRadius: 0,
            borderWidth: 1.6,
            tension: 0,
          };
        }
        trackCompareChart(
          new Chart(execCanvas, {
            type: "line",
            data: {
              labels: labels,
              datasets: [
                execSeries(rule, "规则执行", ARM_COLORS.rule, [4, 3]),
                execSeries(ppo, "原PPO执行", ARM_COLORS.original_ppo),
                execSeries(hitl, "HITL执行", ARM_COLORS.hitl_ppo),
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 10 } } } },
              scales: {
                x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
                y: { title: { display: true, text: "执行量 mm（灌+ / 排−）" } },
              },
            },
            plugins: [editPlug],
          })
        );
      }
      var yr = j.year_row || {};
      var p = yr.original_ppo || {};
      var h = yr.hitl_ppo || {};
      if ($("compareProcessNote")) {
        $("compareProcessNote").textContent =
          year +
          (j.year_role && j.year_role.short ? "（" + j.year_role.short + "）" : "") +
          " 汇总：灌 " +
          fmt(p.irrigation_total, 1) +
          "→" +
          fmt(h.irrigation_total, 1) +
          " mm · 排 " +
          fmt(p.drainage_total, 1) +
          "→" +
          fmt(h.drainage_total, 1) +
          " · 灌水日数 " +
          fmt(p.irrigation_times, 0) +
          "→" +
          fmt(h.irrigation_times, 0) +
          " · 排水日数 " +
          fmt(p.drainage_times, 0) +
          "→" +
          fmt(h.drainage_times, 0) +
          " · 胁迫 " +
          fmt(p.stress_days, 0) +
          "→" +
          fmt(h.stress_days, 0) +
          "（执行量）" +
          (editDays.length
            ? " · 专家修改日 " + editDays.length + "（红竖线）"
            : j.year_role && j.year_role.is_review
              ? " · 本审阅年无改动日标记"
              : " · 留出年无专家改动");
      }
    } catch (e) {
      if ($("compareProcessNote"))
        $("compareProcessNote").textContent = String(e.message || e);
    }
  }

  function renderStageTableHtml(payload) {
    if (!payload || !payload.ok) {
      return '<p class="hint">' + esc((payload && payload.error) || "无生育期数据") + "</p>";
    }
    var ppo = (payload.arms && payload.arms.original_ppo) || [];
    var hitl = (payload.arms && payload.arms.hitl_ppo) || [];
    var byP = {};
    ppo.forEach(function (r) {
      byP[r.stage] = r;
    });
    var byH = {};
    hitl.forEach(function (r) {
      byH[r.stage] = r;
    });
    var stages = [];
    ppo.concat(hitl).forEach(function (r) {
      if (stages.indexOf(r.stage) < 0) stages.push(r.stage);
    });
    var rows = stages
      .map(function (st) {
        var a = byP[st] || {};
        var b = byH[st] || {};
        return (
          "<tr><td>" +
          esc(a.stage_cn || b.stage_cn || st) +
          "</td><td>" +
          fmt(a.irrigation_total, 1) +
          "→" +
          fmt(b.irrigation_total, 1) +
          "</td><td>" +
          fmt(a.drainage_total, 1) +
          "→" +
          fmt(b.drainage_total, 1) +
          "</td><td>" +
          fmt(a.irrigation_times, 0) +
          "→" +
          fmt(b.irrigation_times, 0) +
          "</td><td>" +
          fmt(a.drainage_times, 0) +
          "→" +
          fmt(b.drainage_times, 0) +
          "</td><td>" +
          fmt(a.operation_days, 0) +
          "→" +
          fmt(b.operation_days, 0) +
          "</td><td>" +
          fmt(a.h_before_mean, 1) +
          "→" +
          fmt(b.h_before_mean, 1) +
          "</td><td>" +
          fmt(a.stress_days, 0) +
          "→" +
          fmt(b.stress_days, 0) +
          "</td><td>" +
          fmt(a.below_hmin_days, 0) +
          "→" +
          fmt(b.below_hmin_days, 0) +
          "</td></tr>"
        );
      })
      .join("");
    return (
      "<p class='hint' style='margin:0 0 0.35rem'>" +
      esc(payload.scope_label || "") +
      " · 箭头左侧原PPO、右侧HITL。胁迫按 ks；低于 hmin 另列。</p>" +
      '<table class="compare-table"><thead><tr>' +
      "<th>生育期</th><th>灌水量</th><th>排水量</th><th>灌水日数</th><th>排水日数</th><th>操作日数</th><th>晨水层均值</th><th>胁迫日数</th><th>低于hmin日数</th>" +
      "</tr></thead><tbody>" +
      rows +
      "</tbody></table>"
    );
  }

  async function mountStageTable() {
    var sel = $("compareStageScope");
    var box = $("compareStageTable");
    if (!box) return;
    var raw = sel ? sel.value : "year:2016";
    var url =
      raw === "holdout"
        ? "/api/compare/stage?scope=holdout"
        : "/api/compare/stage?year=" +
          encodeURIComponent(raw.split(":")[1] || "2016");
    try {
      var j = await compareFetchJson(url);
      box.innerHTML = renderStageTableHtml(j);
    } catch (e) {
      box.innerHTML = '<p class="hint">' + esc(String(e.message || e)) + "</p>";
    }
  }

  async function mountAbsorbCharts() {
    var sumEl = $("compareAbsorbSummary");
    try {
      var j = await compareFetchJson("/api/compare/absorption_probs");
      if (!j.ok) {
        if (sumEl) sumEl.textContent = j.error || "吸收概率不可用";
        return;
      }
      var se = j.summary_edited || {};
      var sk = j.summary_confirmed || {};
      var abs = j.absorption || {};
      if (sumEl) {
        sumEl.textContent =
          "改动日 n=" +
          (se.n || 0) +
          " · 专家动作均概率 " +
          fmt(se.mean_ppo_prob, 3) +
          "→" +
          fmt(se.mean_hitl_prob, 3) +
          " · 首选复现率(argmax) " +
          pctTxt(abs.edit_reproduce_rate) +
          "；确认日 n=" +
          (sk.n || 0) +
          " · 原动作均概率 " +
          fmt(sk.mean_ppo_prob, 3) +
          "→" +
          fmt(sk.mean_hitl_prob, 3) +
          " · 保持率 " +
          pctTxt(abs.confirm_hold_rate_policy);
      }
      function drawEditDayList(rows) {
        var box = $("cmpAbsorbEditList");
        if (!box) return;
        var sorted = (rows || []).slice().sort(function (a, b) {
          return toYmd(a.date).localeCompare(toYmd(b.date));
        });
        if (!sorted.length) {
          box.innerHTML = '<p class="hint" style="padding:0.5rem">无改动日样本</p>';
          return;
        }
        var html = sorted
          .map(function (r) {
            var d = toYmd(r.date);
            var pp = Number(r.ppo_prob) || 0;
            var hp = Number(r.hitl_prob) || 0;
            var delta = hp - pp;
            var dCls =
              delta > 1e-3 ? "up" : delta < -1e-3 ? "down" : "flat";
            var dTxt =
              (delta >= 0 ? "+" : "") + delta.toFixed(3);
            var pw = Math.max(0, Math.min(100, pp * 100));
            var hw = Math.max(0, Math.min(100, hp * 100));
            return (
              '<div class="absorb-edit-row" title="' +
              esc(d) +
              " · PPO " +
              pp.toFixed(3) +
              " -> HITL " +
              hp.toFixed(3) +
              '">' +
              '<div class="ae-date">' +
              esc(d) +
              "</div>" +
              '<div class="ae-bars">' +
              '<div class="ae-track"><div class="ae-fill ae-fill-ppo" style="width:' +
              pw.toFixed(1) +
              '%"></div></div>' +
              '<div class="ae-track"><div class="ae-fill ae-fill-hitl" style="width:' +
              hw.toFixed(1) +
              '%"></div></div>' +
              "</div>" +
              '<div class="ae-nums">' +
              '<span class="ae-p">' +
              pp.toFixed(2) +
              "</span>" +
              '<span class="ae-h">' +
              hp.toFixed(2) +
              "</span>" +
              '<span class="ae-d ' +
              dCls +
              '">' +
              dTxt +
              "</span>" +
              "</div></div>"
            );
          })
          .join("");
        box.innerHTML = html;
      }

      function binIndex(p, nBins) {
        if (p == null || !Number.isFinite(p)) return -1;
        if (p >= 1) return nBins - 1;
        if (p <= 0) return 0;
        return Math.min(nBins - 1, Math.floor(p * nBins));
      }

      function drawConfirmDensity(rows) {
        var canvas = $("cmpAbsorbKeepChart");
        var deltaCanvas = $("cmpAbsorbKeepDeltaChart");
        var hint = $("compareAbsorbKeepHint");
        var list = rows || [];
        var n = list.length;
        if (!n) {
          if (hint) hint.textContent = "无确认日样本";
          return;
        }
        var nBins = 10;
        var ppoCounts = [];
        var hitlCounts = [];
        var labels = [];
        var i;
        for (i = 0; i < nBins; i++) {
          ppoCounts.push(0);
          hitlCounts.push(0);
          labels.push(
            (i / nBins).toFixed(1) + "–" + ((i + 1) / nBins).toFixed(1)
          );
        }
        var nNear1 = 0;
        var nDrop = 0;
        var nRise = 0;
        var nFlat = 0;
        var deltas = [];
        list.forEach(function (r) {
          var pp = Number(r.ppo_prob);
          var hp = Number(r.hitl_prob);
          var bi = binIndex(pp, nBins);
          var bj = binIndex(hp, nBins);
          if (bi >= 0) ppoCounts[bi] += 1;
          if (bj >= 0) hitlCounts[bj] += 1;
          if (pp >= 0.9) nNear1 += 1;
          var d = hp - pp;
          deltas.push(d);
          if (d < -1e-3) nDrop += 1;
          else if (d > 1e-3) nRise += 1;
          else nFlat += 1;
        });
        if (hint) {
          hint.textContent =
            "n=" +
            n +
            " · 原PPO≥0.9 占比 " +
            ((100 * nNear1) / n).toFixed(0) +
            "% · Δ≈0 " +
            nFlat +
            " · Δ>0 " +
            nRise +
            " · Δ<0 " +
            nDrop +
            "（分箱柱=样本数，非逐点）";
        }
        if (canvas && typeof Chart !== "undefined") {
          destroyChartOnCanvas(canvas);
          trackCompareChart(
            new Chart(canvas, {
              type: "bar",
              data: {
                labels: labels,
                datasets: [
                  {
                    label: "原PPO",
                    data: ppoCounts,
                    backgroundColor: "rgba(26,122,69,0.7)",
                    borderColor: ARM_COLORS.original_ppo,
                    borderWidth: 1,
                    borderRadius: 2,
                  },
                  {
                    label: "HITL",
                    data: hitlCounts,
                    backgroundColor: "rgba(196,92,38,0.75)",
                    borderColor: ARM_COLORS.hitl_ppo,
                    borderWidth: 1,
                    borderRadius: 2,
                  },
                ],
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                  legend: {
                    position: "top",
                    align: "end",
                    labels: { boxWidth: 12, font: { size: 11 } },
                  },
                  tooltip: {
                    callbacks: {
                      title: function (items) {
                        return items.length
                          ? "概率区间 " + items[0].label
                          : "";
                      },
                      label: function (ctx) {
                        var c = Number(ctx.parsed.y) || 0;
                        return (
                          ctx.dataset.label +
                          ": " +
                          c +
                          " 日（" +
                          ((100 * c) / n).toFixed(1) +
                          "%）"
                        );
                      },
                    },
                  },
                },
                scales: {
                  x: {
                    title: { display: true, text: "动作概率分箱" },
                    ticks: { font: { size: 10 }, maxRotation: 0 },
                    grid: { display: false },
                  },
                  y: {
                    beginAtZero: true,
                    title: { display: true, text: "日数" },
                    ticks: { precision: 0, font: { size: 10 } },
                    grid: { color: "#eef2ef" },
                  },
                },
                datasets: {
                  bar: { categoryPercentage: 0.8, barPercentage: 0.9 },
                },
              },
            })
          );
        }
        if (deltaCanvas && typeof Chart !== "undefined") {
          var dLabels = [
            "≤-0.2",
            "-0.2…-0.1",
            "-0.1…-0.05",
            "-0.05…-0.02",
            "-0.02…0",
            "≈0",
            "0…0.02",
            "0.02…0.05",
            "0.05…0.1",
            "0.1…0.2",
            "≥0.2",
          ];
          var dCounts = dLabels.map(function () {
            return 0;
          });
          function deltaBin(d) {
            if (d <= -0.2) return 0;
            if (d <= -0.1) return 1;
            if (d <= -0.05) return 2;
            if (d <= -0.02) return 3;
            if (d < -0.005) return 4;
            if (d <= 0.005) return 5;
            if (d < 0.02) return 6;
            if (d < 0.05) return 7;
            if (d < 0.1) return 8;
            if (d < 0.2) return 9;
            return 10;
          }
          deltas.forEach(function (d) {
            dCounts[deltaBin(d)] += 1;
          });
          var dColors = dLabels.map(function (_, ix) {
            if (ix < 5) return "rgba(47,111,237,0.75)";
            if (ix === 5) return "rgba(120,120,120,0.55)";
            return "rgba(196,92,38,0.75)";
          });
          destroyChartOnCanvas(deltaCanvas);
          trackCompareChart(
            new Chart(deltaCanvas, {
              type: "bar",
              data: {
                labels: dLabels,
                datasets: [
                  {
                    label: "日数",
                    data: dCounts,
                    backgroundColor: dColors,
                    borderWidth: 0,
                    borderRadius: 2,
                  },
                ],
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { display: false },
                  tooltip: {
                    callbacks: {
                      label: function (ctx) {
                        var c = Number(ctx.parsed.y) || 0;
                        return (
                          c +
                          " 日（" +
                          ((100 * c) / n).toFixed(1) +
                          "%）"
                        );
                      },
                    },
                  },
                },
                scales: {
                  x: {
                    ticks: { font: { size: 9 }, maxRotation: 40, minRotation: 0 },
                    grid: { display: false },
                    title: { display: true, text: "Δ 概率" },
                  },
                  y: {
                    beginAtZero: true,
                    ticks: { precision: 0, font: { size: 10 } },
                    title: { display: true, text: "日数" },
                    grid: { color: "#eef2ef" },
                  },
                },
              },
            })
          );
        }
      }

      drawEditDayList(j.edited || []);
      drawConfirmDensity(j.confirmed || []);
    } catch (e) {
      if (sumEl) sumEl.textContent = String(e.message || e);
    }
  }

  async function mountCompareViz(years, pairsPayload) {
    destroyCompareCharts();
    mountPairCharts(pairsPayload);
    var py = $("compareProcessYear");
    var y0 = (py && py.value) || "2016";
    if (py && !py.dataset.bound) {
      py.dataset.bound = "1";
      py.addEventListener("change", async function () {
        destroyChartOnCanvas("cmpRainChart");
        destroyChartOnCanvas("cmpDepthChart");
        destroyChartOnCanvas("cmpExecChart");
        await mountProcessCharts(py.value || "2016");
      });
    }
    var hm = $("compareHeatMetric");
    if (hm && !hm.dataset.bound) {
      hm.dataset.bound = "1";
      hm.addEventListener("change", function () {
        mountHeatmap();
      });
    }
    var ss = $("compareStageScope");
    if (ss && !ss.dataset.bound) {
      ss.dataset.bound = "1";
      ss.addEventListener("change", function () {
        mountStageTable();
      });
    }
    // 热图 / 过程 / 生育期 / 吸收 并行拉取（吸收最重，与其它同时跑）
    await Promise.all([
      mountHeatmap(),
      mountProcessCharts(y0),
      mountStageTable(),
      mountAbsorbCharts(),
    ]);
  }

  async function renderComparePanel() {
    var gen = ++comparePanelGen;
    var box = $("compareBox");
    if (!box) return;
    destroyCompareCharts();
    var bits = [];
    var year = $("compareYearFilter") ? $("compareYearFilter").value : "";

    var compareMeans = null;
    var compareYearRow = null;
    var pairsPayload = null;
    var yearsForViz = [];
    if (sessionState.has_compare || sessionState.task_compare) {
      try {
        var q = year ? "?year=" + encodeURIComponent(year) : "";
        var j = await fetch("/api/compare/yearly" + q).then(function (r) {
          return r.json();
        });
        if (gen !== comparePanelGen) return;
        if (j.ok) {
          fillCompareYearSelect(j.years || [], j.year_roles || null);
          compareMeans = j.means || null;
          compareYearRow = j.year_row || null;
          pairsPayload = j.pairs || null;
          yearsForViz = j.years || [];
          if (year && j.year_row) {
            var roleTag =
              (j.year_role && (j.year_role.short || j.year_role.label)) ||
              yearRoleShort(year, j.year_roles);
            bits.push(
              renderThreeWayYearRow(j.year_row, year) +
                (roleTag
                  ? '<p class="hint">本视图年角色：' + esc(roleTag) + "（审阅年同样参与评估对比）</p>"
                  : "")
            );
          } else {
            bits.push("<h3 style='margin-top:0'>三方对比 · 多年均值（全部评估年）</h3>");
            bits.push(
              renderThreeWayTable(
                j.task_compare || sessionState.task_compare || {},
                j.means || null
              )
            );
          }
          bits.push(compareVizShellHtml(yearsForViz, pairsPayload));
        } else if (sessionState.task_compare) {
          bits.push("<h3 style='margin-top:0'>三方对比</h3>");
          bits.push(renderThreeWayTable(sessionState.task_compare));
        }
      } catch (e) {
        if (gen !== comparePanelGen) return;
        if (sessionState.task_compare) {
          bits.push(renderThreeWayTable(sessionState.task_compare));
        }
      }
    } else if (lastRunCompare) {
      bits.push("<h3 style='margin-top:0'>训评评估（训练会话 · 尚未三方）</h3>");
      bits.push(renderRunCompareTable(lastRunCompare));
    }

    bits.push(
      renderBcQualityHtml(sessionState.bc_dashboard || {}, {
        year: year,
        means: compareMeans,
        yearRow: compareYearRow,
      })
    );
    if (sessionState.task_compare && sessionState.task_compare.holdout_compare) {
      var ho = sessionState.task_compare.holdout_compare;
      bits.push(
        "<p class='hint'>留出验证年：" +
          esc((ho.holdout_years || []).join(", ")) +
          "（人工审阅默认未改；用于看行为克隆泛化）</p>"
      );
    }
    if (gen !== comparePanelGen) return;
    box.innerHTML = bits.length
      ? bits.join("")
      : '<p class="hint">训评或三方对比完成后显示指标表。</p>';

    if (pairsPayload && pairsPayload.ok) {
      await mountCompareViz(yearsForViz, pairsPayload);
    }
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startPoll() {
    stopPoll();
    pollTimer = setInterval(async function () {
      try {
        var progR = await fetch("/api/run/progress").then(function (r) {
          return r.json();
        });
        if (progR.progress) {
          updateTrainMetrics(progR.progress);
          if (progR.progress.episode != null) {
            var act =
              progR.progress.actions != null
                ? progR.progress.actions
                : progR.progress.action_count;
            livePoints.push({
              episode: progR.progress.episode,
              return: progR.progress.return,
              mean_return: progR.progress.mean_return,
              action_count: act,
            });
            if (livePoints.length > 400) livePoints = livePoints.slice(-400);
            updateTrainChart(livePoints);
          }
        }
        if (progR.busy === false) {
          stopPoll();
          setBusy(false);
          var err =
            progR.error ||
            (progR.progress && progR.progress.error) ||
            (progR.run && progR.run.train_error) ||
            null;
          if (err) {
            setMsg(String(err), true);
            if (progR.progress) updateTrainMetrics(progR.progress);
            else if ($("tmStatus")) $("tmStatus").textContent = "失败";
          } else {
            setMsg("训练/评估完成");
            if (progR.progress) updateTrainMetrics(progR.progress);
          }
          applyRun(progR.run || runState);
          await refreshSummary();
          setStepper(3);
        }
      } catch (e) {
        /* 短暂网络抖动忽略 */
      }
    }, 1500);
  }

  async function refreshSummary() {
    try {
      var j = await fetch("/api/run/summary").then(function (r) {
        return r.json();
      });
      if (j.ok && j.summary) renderSummary(j.summary);
      else renderSummary(null);
      var log = await fetch("/api/run/train_log").then(function (r) {
        return r.json();
      });
      // 无论有无日志都覆盖曲线，避免残留上一会话
      if (log.ok) updateTrainChart(log.rows || [], log.loss_rows || []);
      else updateTrainChart([], []);
    } catch (e) {
      updateTrainChart([], []);
    }
  }

  async function onNewRun() {
    entryChoice = "new";
    setBusy(true);
    setMsg("正在新建 run…");
    try {
      // 默认不拷贝情景表；须手传或点「使用默认情景表」后再校验
      var j = await post("/api/run/new", { snapshot_scenario: false });
      pickingEntry = false;
      applyRun(j.run || {}, { reset: true });
      setMsg(
        "已新建 " +
          ((j.run && j.run.run_id) || "") +
          " · uploads 为空。请「保存上传文件」或点「使用默认情景表」，再点「② 数据集校验」；通过后「开始运行」才会亮起。"
      );
      await refreshRunSelects();
      await refreshSummary();
      setStepper(1);
      scrollToId("card-upload");
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onUseScenario() {
    if (!runState.run_id) {
      setMsg("请先新建或加载训练会话", true);
      return;
    }
    setBusy(true);
    setMsg("正在写入默认情景表…");
    try {
      // 必须带 run_id：页面刷新后服务端可能丢了进程内 _RUN 绑定
      var j = await post("/api/run/snapshot_scenario", {
        run_id: runState.run_id,
      });
      applyRun(j.run || {});
      var snap = j.snapshot || {};
      setMsg(
        (snap.note || "已写入默认情景表") +
          (snap.files && snap.files.length
            ? " · 文件：" + snap.files.join(", ")
            : "") +
          " · 请点「② 数据集校验」"
      );
      setStepper(1);
      scrollToId("card-upload");
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onSwitchRun() {
    var rid = $("runSelectHist") && $("runSelectHist").value;
    if (!rid) return;
    entryChoice = "load";
    setBusy(true);
    try {
      var j = await post("/api/run/switch", { run_id: rid });
      pickingEntry = false;
      applyRun(j.run || {}, { reset: true });
      setMsg(
        "已加载 " +
          rid +
          (j.run && j.run.has_eval
            ? " · 可回放训评结果"
            : j.run && j.run.has_best
              ? " · 有权重，可评估/审阅"
              : " · 新建路径请先校验")
      );
      await refreshSummary();
      setStepper(inferStep());
      scrollToId(
        j.run && (j.run.has_best || j.run.has_eval) ? "card-train" : "card-upload"
      );
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onLoadRun() {
    var rid = $("runSelectHist") && $("runSelectHist").value;
    if (!rid) {
      setMsg("请先在下拉中选择历史训练会话", true);
      return;
    }
    await onSwitchRun();
  }

  async function onLoadSession() {
    var sid = $("sessionSelect") && $("sessionSelect").value;
    if (!sid) {
      setMsg("请先在下拉中选择历史微调会话", true);
      return;
    }
    expertChoice = "load";
    pickingExpert = false;
    await onSwitchSession();
  }

  async function onUpload() {
    if (!runState.run_id) {
      setMsg("请先新建或加载训练会话", true);
      return;
    }
    var map = [
      ["forecast", "upForecast", "训练预报"],
      ["actual", "upActual", "训练实测"],
      ["forecast_eval", "upForecastEval", "评估预报"],
      ["actual_eval", "upActualEval", "评估实测"],
      ["growth_para", "upGrowth", "生育期"],
    ];
    var nPick = 0;
    var badExt = [];
    map.forEach(function (pair) {
      var inp = $(pair[1]);
      if (!(inp && inp.files && inp.files[0])) return;
      nPick += 1;
      var name = String(inp.files[0].name || "");
      if (!/\.csv$/i.test(name)) badExt.push(pair[2] + "（" + name + "）须为 .csv");
    });
    var mdl = $("upModel");
    if (mdl && mdl.files && mdl.files[0]) {
      nPick += 1;
      var mn0 = String(mdl.files[0].name || "");
      if (!/\.pth$/i.test(mn0) && !/\.pt$/i.test(mn0))
        badExt.push("已训模型（" + mn0 + "）须为 .pth / .pt");
    }
    if (nPick === 0) {
      setMsg(
        "未选择任何文件。请在各槽位选择 .csv 后再点「保存上传文件」；" +
          "若要用情景默认表，请点「使用默认情景表」。" +
          "下方灰色标签是磁盘已有文件，不是本次上传结果。",
        true
      );
      return;
    }
    if (badExt.length) {
      setMsg(badExt.join("；"), true);
      return;
    }
    setBusy(true);
    setMsg("上传中…");
    try {
      var fd = new FormData();
      var mn = ($("modelName") && $("modelName").value) || "";
      if (mn) fd.append("model_name", mn);
      fd.append("run_id", runState.run_id);
      map.forEach(function (pair) {
        var inp = $(pair[1]);
        if (inp && inp.files && inp.files[0]) fd.append(pair[0], inp.files[0]);
      });
      if (mdl && mdl.files && mdl.files[0]) fd.append("model", mdl.files[0]);
      var r = await fetch("/api/run/upload", { method: "POST", body: fd });
      var j = await r.json();
      if (!r.ok || j.ok === false) throw new Error(j.error || "上传失败");
      applyRun(j.run || {});
      setMsg(
        "本次已保存：" +
          ((j.saved && j.saved.length ? j.saved.join(", ") : "（无）")) +
          (j.model_saved ? "；模型 " + j.model_saved : "") +
          " · 请重新点「② 数据集校验」"
      );
      setStepper(1);
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onValidate() {
    if (!runState.run_id) {
      setMsg("请先新建或加载训练会话", true);
      return;
    }
    setBusy(true);
    setMsg("校验中…");
    try {
      var mode =
        ($("trainMode") && $("trainMode").value) === "eval_only" ? "eval_only" : "full";
      var j = await post("/api/run/validate", {
        mode: mode,
        run_id: runState.run_id,
      });
      applyRun(j.run || {});
      setMsg(j.validate && j.validate.ok ? "校验通过" : "校验未通过", !(j.validate && j.validate.ok));
      if (j.validate && j.validate.ok) setStepper(2);
      updateTrainGate();
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onStartTrain() {
    if (!runState.run_id) {
      setMsg("请先新建或加载训练会话", true);
      return;
    }
    setBusy(true);
    livePoints = [];
    setMsg("启动训练/评估…");
    setStepper(3);
    try {
      var body = {
        mode: ($("trainMode") && $("trainMode").value) || "auto",
        episodes: Number(($("trainEpisodes") && $("trainEpisodes").value) || 1500),
        model_name: ($("modelName") && $("modelName").value) || "ppo_rice",
        run_id: runState.run_id,
      };
      var my = $("maxYearsTrain") && $("maxYearsTrain").value;
      if (my) body.max_years = Number(my);
      await post("/api/run/start_train", body);
      startPoll();
      setMsg("训练进行中…");
    } catch (e) {
      setBusy(false);
      setMsg(String(e.message || e), true);
    }
  }

  async function refreshRunSelects() {
    try {
      var boot = await fetch("/api/boot").then(function (r) {
        return r.json();
      });
      var runs = boot.runs || [];
      var sessions = boot.sessions || [];
      var hist = $("runSelectHist");
      if (hist) {
        var cur = runState.run_id || "";
        hist.innerHTML =
          '<option value="">— 选择历史训练会话 —</option>' +
          runs
            .map(function (r) {
              return (
                '<option value="' +
                esc(r.run_id) +
                '"' +
                (r.run_id === cur ? " selected" : "") +
                ">" +
                esc(r.run_id) +
                (r.has_best ? " ·已训" : "") +
                (r.has_eval ? " ·已评" : "") +
                "</option>"
              );
            })
            .join("");
      }
      var exp = $("expertRunSelect");
      if (exp) {
        var curE = runState.run_id || "";
        exp.innerHTML =
          '<option value="">— 请选择含 best 的训练会话 —</option>' +
          runs
            .filter(function (r) {
              return r.has_best;
            })
            .map(function (r) {
              return (
                '<option value="' +
                esc(r.run_id) +
                '"' +
                (r.run_id === curE ? " selected" : "") +
                ">" +
                esc(r.run_id) +
                (r.has_eval ? " ·已评估" : "") +
                "</option>"
              );
            })
            .join("");
      }
      var ss = $("sessionSelect");
      if (ss) {
        var curS = sessionState.session_id || "";
        ss.innerHTML =
          '<option value="">— ' +
          (curS ? "当前微调会话 " + esc(curS) : "选择历史微调会话") +
          " —</option>" +
          sessions
            .map(function (s) {
              return (
                '<option value="' +
                esc(s.session_id) +
                '"' +
                (s.session_id === curS ? " selected" : "") +
                ">" +
                esc(s.session_id) +
                (s.run_id ? " · " + esc(s.run_id) : "") +
                "</option>"
              );
            })
            .join("");
      }
    } catch (e) {}
  }

  async function gotoExpert() {
    setStepper(4);
    scrollToId("card-expert");
    if (runState.run_id && $("expertRunSelect")) {
      $("expertRunSelect").value = runState.run_id;
    }
  }

  function markHtml(r) {
    var bits = [];
    if (r.needs_reconfirm)
      bits.push(
        '<span class="tag warn" title="上游改动后重推建议，请看水深是否合理；要改就改专家动作">需重看</span>'
      );
    if (r.needs_reconfirm && r.editable)
      bits.push(
        '<button type="button" class="tag ok tag-btn confirm-proposal" data-i="' +
          esc(r.i) +
          '" title="确认重推后的当前动作；不改变这一天的执行轨迹">认可当前建议</button>'
      );
    else if (r.confirmed && r.editable)
      bits.push('<span class="tag ok" title="该日动作已由专家确认">已确认</span>');
    var expertSet =
      r.a_expert != null && r.a_expert !== "" && !isNaN(Number(r.a_expert));
    if (expertSet && r.editable)
      bits.push('<span class="tag ok" title="专家已指定动作（含与模型相同档位）">已填</span>');
    if (
      expertSet &&
      r.a_model != null &&
      Math.abs(Number(r.a_expert) - Number(r.a_model)) > 1e-6 &&
      r.editable
    )
      bits.push('<span class="tag ok" title="专家动作≠模型提案">已改</span>');
    if (r.prefer_review) {
      var tip =
        "分歧=模型执行≠经验规则。点击可开关筛选。提案 " +
        esc(r.a_model) +
        " · 执行 " +
        fmt(r.a_exec != null ? r.a_exec : r.a_applied, 0) +
        " · 规则 " +
        fmtActionMm(r.a_rule);
      bits.push(
        '<button type="button" class="tag warn tag-btn prefer-tag" title="' +
          tip +
          '">与规则分歧</button>'
      );
    }
    if (r.is_small_run)
      bits.push(
        '<span class="tag warn" title="连续小额灌溉（执行量）：全日 0&lt;m≤20mm 且≥2天，或续灌后的小额补灌日；排水另计">连续小额灌</span>'
      );
    if (!r.editable || r.used_fixed_rule) {
      var fl = r.fixed_label || (r.used_fixed_rule ? "固定规则" : "固定");
      bits.push(
        '<span class="tag" title="固定规则日：自动执行">' + esc(fl) + "</span>"
      );
    }
    if (r.used_hc_guard)
      bits.push('<span class="tag warn" title="超过 Hc 被安全守卫改写">Hc守卫</span>');
    return bits.length ? bits.join(" ") : "—";
  }

  function dateOrdKey(d) {
    var s = String(d || "").replace(/-/g, "/").split(" ")[0];
    var p = s.split("/");
    if (p.length !== 3) return [0, 0, 0];
    return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
  }

  function sortRowsByDate(rows) {
    return (rows || []).slice().sort(function (a, b) {
      var ya = String(a.year || "");
      var yb = String(b.year || "");
      if (ya !== yb) return ya < yb ? -1 : 1;
      var ka = dateOrdKey(a.date);
      var kb = dateOrdKey(b.date);
      for (var i = 0; i < 3; i++) {
        if (ka[i] !== kb[i]) return ka[i] - kb[i];
      }
      return (a.i || 0) - (b.i || 0);
    });
  }

  function expertFilled(row) {
    return row.a_expert != null && row.a_expert !== "" && !isNaN(Number(row.a_expert));
  }

  function expertSelect(r) {
    var hasOv = Object.prototype.hasOwnProperty.call(expertOverrides, r.i);
    var cur = null;
    if (hasOv) {
      if (expertOverrides[r.i] === null || expertOverrides[r.i] === "") cur = null;
      else cur = Number(expertOverrides[r.i]);
    } else if (expertFilled(r)) cur = Number(r.a_expert);
    var opts =
      '<option value=""' +
      (cur === null ? " selected" : "") +
      ">—</option>";
    opts += (r.allowed_actions || ACTION_SPACE).map(function (a) {
      return (
        '<option value="' +
        a +
        '"' +
        (cur !== null && Number(a) === cur ? " selected" : "") +
        ">" +
        a +
        "</option>"
      );
    }).join("");
    return (
      '<select data-i="' +
      r.i +
      '" class="expertSel"' +
      (r.editable ? "" : " disabled") +
      ">" +
      opts +
      "</select>"
    );
  }

  function reasonInput(r) {
    var hasCode = Object.prototype.hasOwnProperty.call(expertReasonCodes, r.i);
    var code = hasCode
      ? String(expertReasonCodes[r.i] || "")
      : String(r.reason_code || "");
    if (!hasCode && code) expertReasonCodes[r.i] = code;
    var hasText = Object.prototype.hasOwnProperty.call(expertReasons, r.i);
    var text = hasText ? String(expertReasons[r.i] || "") : String(r.reason || "");
    if (!hasText && text) expertReasons[r.i] = text;
    var opts =
      '<option value="">—改因—</option>' +
      REASON_CODE_OPTIONS.map(function (o) {
        return (
          '<option value="' +
          esc(o.code) +
          '"' +
          (code === o.code ? " selected" : "") +
          ">" +
          esc(o.label) +
          "</option>"
        );
      }).join("");
    return (
      '<div class="reason-cell">' +
      '<select class="reasonCode" data-i="' +
      esc(r.i) +
      '"' +
      (r.editable ? "" : " disabled") +
      ">" +
      opts +
      "</select>" +
      '<input type="text" class="reason" data-i="' +
      esc(r.i) +
      '" value="' +
      esc(text) +
      '" placeholder="补充说明（可选）"' +
      (r.editable ? "" : " disabled") +
      " />" +
      "</div>"
    );
  }

  function updateWaterChart(rows) {
    ensureWaterCharts();
    if (!waterChart) return;
    var list = rows || [];
    function col(key) {
      return list.map(function (r) {
        return r[key];
      });
    }
    var dateLabels = list.map(function (r) {
      return formatDateOnly(r.date);
    });
    applyDepthAxisRange(waterChart, list);
    waterChart.data.labels = dateLabels;
    waterChart.data.datasets = [
      {
        type: "line",
        label: "水深 h",
        data: col("h_before"),
        yAxisID: "y",
        borderColor: "#1a4d32",
        backgroundColor: "rgba(45,90,66,0.06)",
        borderWidth: 2.6,
        pointRadius: list.map(function (r) {
          return r.prefer_review ? 3.5 : 0;
        }),
        pointBackgroundColor: list.map(function (r) {
          return r.prefer_review ? "#9a5b12" : "#1a4d32";
        }),
        tension: 0.12,
        order: 0,
      },
    ].concat(
      buildThresholdDatasets(function (_src, key) {
        return col(key);
      }, list)
    );
    waterChart.update("none");

    if (waterActionChart) {
      waterActionChart.data.labels = dateLabels;
      waterActionChart.data.datasets = buildIrrigDrainBars(list);
      waterActionChart.update("none");
    }
    if (waterRainChart) {
      waterRainChart.data.labels = dateLabels;
      waterRainChart.data.datasets = buildReviewForecastRainBars(list);
      waterRainChart.update("none");
    }
    if ($("waterRainCaption")) {
      var rains = list.map(forecastRainMm);
      var known = rains.filter(function (v) { return v != null; }).length;
      $("waterRainCaption").textContent =
        "决策时可用的未来7日预报累计雨量 · P_cumu7 (mm) · 有效 " +
        known + "/" + rains.length + " 日（审阅不显示实测日雨）";
    }
  }

  function reviewYearOptions(yearsInTable) {
    // 下拉只列规范审阅年（∩ 表内年）；不含留出验证年
    var table = (yearsInTable || []).map(String);
    var need =
      (sessionState.expected_review_years && sessionState.expected_review_years.length
        ? sessionState.expected_review_years
        : expectedReviewYears) || [];
    need = need.map(String);
    if (need.length) {
      var hit = need.filter(function (y) {
        return table.indexOf(y) >= 0;
      });
      return hit.length ? hit : need;
    }
    return table;
  }

  function refreshYearGapHint(yearsInTable) {
    var el = $("yearGapHint");
    if (!el) return;
    var opts = reviewYearOptions(yearsInTable);
    var need = (expectedReviewYears || []).map(String);
    var table = (yearsInTable || []).map(String);
    var missingInTable = need.filter(function (y) {
      return table.indexOf(y) < 0;
    });
    var missConfirm = sessionState.missing_review_years || [];
    if (missingInTable.length) {
      el.innerHTML =
        "当前审阅表缺规范审阅年 <b>" +
        esc(missingInTable.join(", ")) +
        "</b>（现有 " +
        esc(table.join(", ") || "—") +
        "）。请「更换微调方式」后<strong>留空年数上限</strong>重新生成。";
      setHidden(el, false);
    } else {
      var confBit = missConfirm.length
        ? " · 尚未定稿：" + missConfirm.join(", ") + "（逐年点「本审阅年定稿」后才能 BC）"
        : " · 各审阅年均已定稿";
      el.textContent =
        "审阅年（可切换）：" + opts.join(", ") + confBit + "。留出验证年不进此下拉。";
      setHidden(el, false);
    }
  }

  async function loadRows() {
    // 图始终拉全年（prefer=0）；表格再按「仅分歧日」筛
    var preferOnly = !!($("onlyPrefer") && $("onlyPrefer").checked);
    var year = $("yearFilter") ? $("yearFilter").value : "";
    var url = "/api/rows?prefer=0";
    if (year) url += "&year=" + encodeURIComponent(year);
    var r = await fetch(url);
    var j = await r.json();
    if (!j.ok) return;
    if (j.action_space && j.action_space.length) ACTION_SPACE = j.action_space;
    var ysel = $("yearFilter");
    if (ysel && j.years) {
      var cur = ysel.value;
      var yearOpts = reviewYearOptions(j.years);
      // 下拉只列规范审阅年（不含留出验证年）；强制单年，与三图「当前所选年」一致
      ysel.innerHTML = yearOpts.length
        ? yearOpts
            .map(function (y) {
              return '<option value="' + esc(y) + '">' + esc(y) + " 整季</option>";
            })
            .join("")
        : '<option value="">（无审阅年）</option>';
      if (cur && yearOpts.indexOf(cur) >= 0) {
        ysel.value = cur;
      } else if (yearOpts.length) {
        yearFilterAutoDone = true;
        ysel.value = yearOpts[0];
        if (cur !== yearOpts[0]) return loadRows();
      } else {
        ysel.value = "";
      }
      refreshYearGapHint(j.years);
    }
    lastRowsAll = sortRowsByDate(j.rows || []);
    updateWaterChart(lastRowsAll);
    var showFixed = !!($("showFixedDays") && $("showFixedDays").checked);
    lastRows = lastRowsAll.filter(function (row) {
      if (preferOnly && !row.prefer_review) return false;
      // 取消「显示固定规则日」时才隐藏晒田/落干等锁定行
      if (!showFixed && !row.editable) return false;
      return true;
    });
    var tb = $("reviewTbody");
    if (!tb) return;
    if (!lastRows.length) {
      tb.innerHTML =
        '<tr><td colspan="14" class="empty">无行（可取消「仅分歧」、取消「显示固定规则日」、或换年）</td></tr>';
      return;
    }
    tb.innerHTML = lastRows
      .map(function (row) {
        var cls = [];
        if (row.prefer_review) cls.push("prefer");
        if (!row.editable) cls.push("locked");
        var aRule = "当前 " + fmtActionMm(row.a_rule_current_state) +
          '<br/><small>原轨迹 ' + fmtActionMm(row.a_rule) + '</small>';
        var aExec = fmtActionMm(
          row.a_exec != null ? row.a_exec : row.a_applied
        );
        var rain = forecastRainMm(row);
        var rainCell =
          rain == null || rain === "" || Number.isNaN(Number(rain))
            ? "—"
            : Number(rain) > 0
              ? fmt(rain, 1)
              : "0";
        return (
          '<tr class="' +
          cls.join(" ") +
          (Number(rain) > 0 ? " rain-day" : "") +
          '"><td>' +
          esc(formatDateOnly(row.date)) +
          "</td><td>" +
          esc(row.year) +
          "</td><td>" +
          esc(row.stage) +
          "</td><td>" +
          fmt(row.h_before, 1) +
          "</td><td class=" +
          (Number(rain) > 0 ? "rain-mm" : "") +
          ">" +
          rainCell +
          "</td><td>" +
          fmt(row.hmin, 0) +
          "</td><td>" +
          fmt(row.hmax, 0) +
          "</td><td>" +
          fmt(row.Hp, 0) +
          "</td><td>" +
          fmt(row.Hc, 0) +
          "</td><td>" +
          esc(row.a_model) +
          "</td><td>" +
          aExec +
          "</td><td>" +
          aRule +
          "</td><td>" +
          expertSelect(row) +
          "</td><td>" +
          reasonInput(row) +
          "</td><td>" +
          markHtml(row) +
          "</td></tr>"
        );
      })
      .join("");

    tb.querySelectorAll(".prefer-tag").forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!$("onlyPrefer")) return;
        $("onlyPrefer").checked = !$("onlyPrefer").checked;
        loadRows();
      });
    });
    tb.querySelectorAll(".confirm-proposal").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var i = btn.getAttribute("data-i");
        expertConfirms[i] = true;
        scheduleAutoResync();
      });
    });
    tb.querySelectorAll(".expertSel").forEach(function (sel) {
      sel.addEventListener("change", function () {
        var i = sel.getAttribute("data-i");
        if (sel.value === "") expertOverrides[i] = null;
        else expertOverrides[i] = Number(sel.value);
        var codeSel = tb.querySelector('select.reasonCode[data-i="' + i + '"]');
        var code = codeSel ? String(codeSel.value || "") : String(expertReasonCodes[i] || "");
        if (sel.value !== "" && !code) {
          setMsg(
            "已改专家动作：请同时选择结构化改因（合并小额 / 预报暂缓灌水 / 允许较低水层 / 其他）",
            true
          );
        }
        scheduleAutoResync();
      });
    });
    tb.querySelectorAll("select.reasonCode").forEach(function (sel) {
      sel.addEventListener("change", function () {
        expertReasonCodes[sel.getAttribute("data-i")] = sel.value;
        if (!Object.keys(expertOverrides).length && !Object.keys(expertConfirms).length) {
          scheduleAutoResync();
        }
      });
    });
    tb.querySelectorAll("input.reason").forEach(function (inp) {
      inp.addEventListener("input", function () {
        expertReasons[inp.getAttribute("data-i")] = inp.value;
      });
      inp.addEventListener("change", function () {
        expertReasons[inp.getAttribute("data-i")] = inp.value;
        if (!Object.keys(expertOverrides).length && !Object.keys(expertConfirms).length) {
          scheduleAutoResync();
        }
      });
    });
  }

  function collectPayload() {
    var overrides = {};
    Object.keys(expertOverrides).forEach(function (k) {
      overrides[k] = expertOverrides[k];
    });
    var confirms = {};
    Object.keys(expertConfirms).forEach(function (k) {
      confirms[k] = expertConfirms[k];
    });
    var reasons = {};
    var reason_codes = {};
    Object.keys(expertReasons).forEach(function (k) {
      reasons[k] = String(expertReasons[k] == null ? "" : expertReasons[k]).trim();
    });
    Object.keys(expertReasonCodes).forEach(function (k) {
      reason_codes[k] = String(expertReasonCodes[k] || "").trim();
    });
    document.querySelectorAll("#reviewTbody input.reason").forEach(function (inp) {
      reasons[inp.getAttribute("data-i")] = String(inp.value || "").trim();
    });
    document.querySelectorAll("#reviewTbody select.reasonCode").forEach(function (sel) {
      reason_codes[sel.getAttribute("data-i")] = String(sel.value || "").trim();
    });
    return {
      overrides: overrides,
      confirms: confirms,
      reasons: reasons,
      reason_codes: reason_codes,
    };
  }

  async function onStartExpert() {
    var run =
      ($("expertRunSelect") && $("expertRunSelect").value) || runState.run_id;
    if (!run) {
      setMsg("请选择训练会话", true);
      return;
    }
    expertChoice = "new";
    setBusy(true);
    setMsg("正在生成人工审阅表…");
    setStepper(4);
    try {
      var body = { from_run: run };
      var my = $("maxYearsExpert") && $("maxYearsExpert").value;
      if (my) body.max_years = Number(my);
      var j = await post("/api/start_session", body);
      pickingExpert = false;
      applySession(j.session || {}, { reset: true });
      setMsg(
        "已创建 " +
          ((j.session && j.session.session_id) || "") +
          " · " +
          j.n_days +
          " 天 · 分歧 " +
          j.n_prefer
      );
      await refreshRunSelects();
      scrollToId("card-expert");
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  function onResetEntry() {
    pickingEntry = true;
    // 重选入口时先清空训评回放，避免仍显示上一 run 曲线
    clearRunPlayback();
    boundRunId = null;
    syncBoardVisibility();
    setMsg("请重新选择：A 新建 或 B 加载");
    scrollToId("card-entry");
  }

  function onResetExpert() {
    pickingExpert = true;
    clearSessionPlayback();
    boundSessionId = null;
    sessionState = {};
    syncBoardVisibility();
    setMsg("请重新选择微调方式：① 加载 或 ② 新建人工审阅");
    scrollToId("card-expert");
  }

  var resyncTimer = null;
  var resyncInFlight = false;
  var resyncQueued = false;

  function scheduleAutoResync() {
    markReviewDirty(true);
    if ($("resyncStatus"))
      $("resyncStatus").textContent = "专家动作/确认已更新 → 即将自动一致性重算…";
    if (resyncTimer) clearTimeout(resyncTimer);
    resyncTimer = setTimeout(function () {
      doResync({ auto: true });
    }, 400);
  }

  function clearSubmittedLocal(payload) {
    Object.keys((payload && payload.overrides) || {}).forEach(function (k) {
      delete expertOverrides[k];
    });
    Object.keys((payload && payload.confirms) || {}).forEach(function (k) {
      delete expertConfirms[k];
    });
  }

  async function doResync(opt) {
    opt = opt || {};
    if (resyncInFlight) {
      resyncQueued = true;
      if ($("resyncStatus"))
        $("resyncStatus").textContent = "重算进行中 → 当前改动将排队自动再算…";
      return null;
    }
    var payload = collectPayload();
    var nOv = Object.keys(payload.overrides || {}).length;
    var nCf = Object.keys(payload.confirms || {}).length;
    if (!opt.force && !reviewDirty && nOv === 0 && nCf === 0) {
      if (!opt.auto)
        setMsg("没有待提交的改动", true);
      return null;
    }
    resyncInFlight = true;
    setBusy(true);
    setMsg(
      opt.finalize
        ? "定稿中：确认本审阅年并写入 BC 样本源…"
        : "一致性重算中（按专家动作重滚本季水深与后续日）…"
    );
    try {
      var j = await post("/api/resync", payload);
      clearSubmittedLocal(payload);
      applySession(j.session || {});
      var stillPending =
        Object.keys(expertOverrides).length > 0 || resyncQueued;
      var deferredNote =
        Number(j.n_deferred_actions || 0) > 0
          ? " · 同批后续 " + j.n_deferred_actions + " 个动作因状态变化已作废，请按新建议复核"
          : "";
      markReviewDirty(stillPending);
      setStepper(4);
      if (opt.finalize) {
        setMsg(
          "本审阅年已定稿 · 确认 " +
            j.n_confirmed +
            " 天 · 待复核 " +
            j.n_needs_reconfirm +
            " · 可换年继续；各审阅年就绪后做行为克隆"
        );
        if ($("resyncStatus"))
          $("resyncStatus").textContent =
            "本审阅年已定稿（确认全日可进 BC）。换年继续改动作（自动重算）→ 再定稿。";
      } else {
        setMsg(
          (opt.auto ? "已自动重算 · " : "重算完成 · ") +
            "确认 " +
            j.n_confirmed +
            " · 待复核 " +
            j.n_needs_reconfirm +
            (stillPending ? " · 还有改动排队中…" : "") + deferredNote
        );
        if ($("resyncStatus"))
          $("resyncStatus").textContent = stillPending
            ? "还有未提交改动，将自动再重算…"
            : "一致性已更新；逐日处理「需重看」：认可当前建议或选专家动作，再定稿。" + deferredNote;
      }
      await loadRows();
      return j;
    } catch (e) {
      setMsg(String(e.message || e), true);
      return null;
    } finally {
      resyncInFlight = false;
      setBusy(false);
      var again =
        resyncQueued || Object.keys(expertOverrides).length > 0;
      resyncQueued = false;
      if (again) scheduleAutoResync();
    }
  }

  async function onResync() {
    return doResync({ auto: false, force: true });
  }

  /** 用 original_trajectories 覆盖当前审阅年，丢弃本学年改动 */
  async function onResetYear() {
    var year = $("yearFilter") ? $("yearFilter").value : "";
    if (!year) {
      setMsg("请先选择要重置的审阅年", true);
      return;
    }
    if (resyncInFlight) {
      setMsg("请等待当前重算完成后再重置", true);
      return;
    }
    if (
      !window.confirm(
        "确认将 " +
          year +
          " 年审阅表恢复为 original 原始轨迹？本学年已改专家动作会丢失。"
      )
    ) {
      return;
    }
    setBusy(true);
    setMsg("正在重置 " + year + " 年为原始轨迹…");
    try {
      var j = await post("/api/review/reset_year", { year: year });
      expertOverrides = {};
      expertConfirms = {};
      expertReasons = {};
      expertReasonCodes = {};
      markReviewDirty(false);
      applySession(j.session || {});
      setMsg(
        "已重置 " +
          year +
          " 年 · 恢复 " +
          (j.n_restored || "") +
          " 天（来自 original）"
      );
      if ($("resyncStatus"))
        $("resyncStatus").textContent =
          year + " 年已恢复原始轨迹，可重新审阅。";
      await loadRows();
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  /** 一键认可本审阅年全部需重看（接纳当前 PPO 建议） */
  async function onAcceptPendingYear() {
    var year = $("yearFilter") ? $("yearFilter").value : "";
    if (!year) {
      setMsg("请先选择审阅年", true);
      return;
    }
    if (resyncInFlight) {
      setMsg("请等待当前重算完成", true);
      return;
    }
    if (resyncTimer) {
      clearTimeout(resyncTimer);
      resyncTimer = null;
    }
    if (
      reviewDirty ||
      Object.keys(expertOverrides).length > 0 ||
      Object.keys(expertConfirms).length > 0
    ) {
      var syncResult = await doResync({ force: true });
      if (!syncResult || !syncResult.ok) return;
    }
    setBusy(true);
    setMsg("正在认可 " + year + " 年全部需重看日…");
    try {
      var j = await post("/api/review/accept_pending_year", { year: year });
      expertOverrides = {};
      expertConfirms = {};
      expertReasons = {};
      expertReasonCodes = {};
      markReviewDirty(false);
      applySession(j.session || {});
      setMsg(
        year +
          " 年已认可需重看 " +
          (j.n_accepted || 0) +
          " 天；请再点「本审阅年定稿」纳入 BC"
      );
      if ($("resyncStatus"))
        $("resyncStatus").textContent =
          year + " 需重看已清。请点「本审阅年定稿」；四年都定稿后才能 BC。";
      await loadRows();
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  /** 本审阅年定稿：接纳当前年轨迹（未改=认可）→ 写入 BC 样本源 */
  async function onFinalizeYear() {
    var year = $("yearFilter") ? $("yearFilter").value : "";
    if (!year) {
      setMsg("请先选择审阅年", true);
      return;
    }
    if (resyncInFlight) {
      setMsg("请等待当前重算完成后再定稿", true);
      return;
    }
    if (resyncTimer) {
      clearTimeout(resyncTimer);
      resyncTimer = null;
    }
    if (
      reviewDirty ||
      Object.keys(expertOverrides).length > 0 ||
      Object.keys(expertConfirms).length > 0
    ) {
      var syncResult = await doResync({ force: true });
      if (!syncResult || !syncResult.ok) return;
    }
    setBusy(true);
    setMsg("定稿中：接纳 " + year + " 年当前轨迹…");
    try {
      var j = await post("/api/review/finalize_year", { year: year });
      expertOverrides = {};
      expertConfirms = {};
      expertReasons = {};
      expertReasonCodes = {};
      markReviewDirty(false);
      applySession(j.session || {});
      setMsg(
        "本审阅年已定稿 · " +
          year +
          " · 确认 " +
          (j.n_confirmed || "") +
          " 天 · 改动 " +
          (j.n_edited || 0) +
          " 天"
      );
      if ($("resyncStatus"))
        $("resyncStatus").textContent =
          year + " 已定稿。换年继续，或各年就绪后做行为克隆。";
      await loadRows();
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onBc() {
    if (resyncInFlight || reviewDirty) {
      setMsg("仍有未定稿/未完成的重算：请先完成本审阅年定稿", true);
      scrollToId("card-expert");
      return;
    }
    if (!sessionState.bc_ready) {
      var miss = sessionState.missing_review_years || [];
      var msg =
        Number(sessionState.n_edit || 0) <= 0
          ? "改动日为 0：请先在审阅年改动作并一致性重算后再 BC"
          : miss.length
            ? "审阅年未全部确认（缺 " + miss.join(", ") + "），不能提交 BC"
            : "审阅未就绪，不能提交行为克隆";
      setMsg(msg, true);
      scrollToId("card-expert");
      return;
    }
    setBusy(true);
    setMsg("行为克隆(BC) 训练中…");
    try {
      clearCompareFetchCache();
      var j = await post("/api/bc", {
        epochs: Number(($("bcEpochs") && $("bcEpochs").value) || 80),
        w_edit: Number(($("bcWEdit") && $("bcWEdit").value) || 6),
      });
      clearCompareFetchCache();
      // 对比跑完再统一渲染，避免中途与 /api/metrics 抢写 absorption_probs
      applySession(j.session || {}, { skipComparePanel: true });
      var res = j.result || {};
      setStepper(6);
      scrollToId("card-eval");
      setMsg(
        "行为克隆完成 · 标签 " +
          (res.n_labels || "") +
          " · 改动 " +
          (res.n_edited || "") +
          " · 正在自动跑三方对比…"
      );
      try {
        await runThreeWayCompare({ quiet: true });
        setMsg(
          "行为克隆完成并已自动三方对比 · 标签 " +
            (res.n_labels || "") +
            " · 改动 " +
            (res.n_edited || "")
        );
      } catch (e2) {
        setMsg(
          "行为克隆已完成，但自动三方对比失败：" +
            String(e2.message || e2) +
            "（可手动点「跑三方对比」）",
          true
        );
      }
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  /** 三方对比核心；quiet=true 时不抢占 busy（由外层如 onBc 控制） */
  async function runThreeWayCompare(opt) {
    opt = opt || {};
    setMsg("三方对比中…");
    setStepper(6);
    clearCompareFetchCache();
    var j = await post("/api/compare", {});
    clearCompareFetchCache();
    applySession(j.session || {}, { skipComparePanel: true });
    try {
      await post("/api/metrics", {});
      var boot = await fetch("/api/boot").then(function (r) {
        return r.json();
      });
      if (boot.session) applySession(boot.session, { skipComparePanel: true });
    } catch (e2) {}
    clearCompareFetchCache();
    await renderComparePanel();
    scrollToId("card-eval");
    if (!opt.quiet) setMsg("三方对比完成");
    return j;
  }

  async function onCompare() {
    setBusy(true);
    try {
      clearCompareFetchCache();
      await runThreeWayCompare({});
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  async function onSwitchSession() {
    var sid = $("sessionSelect") && $("sessionSelect").value;
    if (!sid) return;
    expertChoice = "load";
    setBusy(true);
    try {
      var j = await post("/api/switch_session", { session_id: sid });
      pickingExpert = false;
      applySession(j.session || {}, { reset: true });
      setMsg("已加载 " + sid);
      setStepper(inferStep());
      scrollToId("card-expert");
    } catch (e) {
      setMsg(String(e.message || e), true);
    } finally {
      setBusy(false);
    }
  }

  function bind() {
    document.querySelectorAll("#stepper .step").forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (btn.disabled || btn.classList.contains("is-skipped")) return;
        var s = Number(btn.getAttribute("data-step"));
        setStepper(s);
        var target = btn.getAttribute("data-target") || "card-entry";
        scrollToId(target);
      });
    });
    document.querySelectorAll("#entryTabs .mode-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        entryChoice = btn.getAttribute("data-entry") || "new";
        syncBoardVisibility();
      });
    });
    document.querySelectorAll("#expertTabs .mode-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        expertChoice = btn.getAttribute("data-expert") || "load";
        syncBoardVisibility();
      });
    });
    if ($("btnResetEntry")) $("btnResetEntry").addEventListener("click", onResetEntry);
    if ($("btnResetExpert"))
      $("btnResetExpert").addEventListener("click", onResetExpert);
    if ($("btnNewRun")) $("btnNewRun").addEventListener("click", onNewRun);
    if ($("btnLoadRun")) $("btnLoadRun").addEventListener("click", onLoadRun);
    if ($("btnLoadSession"))
      $("btnLoadSession").addEventListener("click", onLoadSession);
    // 下拉仅预选，确认靠按钮（避免未点加载就切换会话）
    if ($("btnUpload")) $("btnUpload").addEventListener("click", onUpload);
    if ($("btnUseScenario"))
      $("btnUseScenario").addEventListener("click", onUseScenario);
    if ($("btnValidate")) $("btnValidate").addEventListener("click", onValidate);
    if ($("btnStartTrain"))
      $("btnStartTrain").addEventListener("click", onStartTrain);
    if ($("btnGotoExpert"))
      $("btnGotoExpert").addEventListener("click", gotoExpert);
    if ($("trainMode"))
      $("trainMode").addEventListener("change", updateTrainGate);
    function syncSmoothSeg() {
      var sel = $("smoothWindow");
      var cur = sel ? String(sel.value) : "100";
      document.querySelectorAll(".smooth-opt").forEach(function (btn) {
        btn.classList.toggle("on", String(btn.getAttribute("data-w")) === cur);
      });
    }
    document.querySelectorAll(".smooth-opt").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var w = btn.getAttribute("data-w") || "0";
        if ($("smoothWindow")) $("smoothWindow").value = w;
        syncSmoothSeg();
        updateProcessCharts(lastLogRows, lastLossRows);
      });
    });
    if ($("smoothWindow")) {
      $("smoothWindow").addEventListener("change", function () {
        syncSmoothSeg();
        updateProcessCharts(lastLogRows, lastLossRows);
      });
      syncSmoothSeg();
    }
    if ($("btnStartExpert"))
      $("btnStartExpert").addEventListener("click", onStartExpert);
    if ($("btnResync")) $("btnResync").addEventListener("click", onResync);
    if ($("btnAcceptPendingYear"))
      $("btnAcceptPendingYear").addEventListener("click", onAcceptPendingYear);
    if ($("btnFinalizeYear"))
      $("btnFinalizeYear").addEventListener("click", onFinalizeYear);
    if ($("btnResetYear"))
      $("btnResetYear").addEventListener("click", onResetYear);
    if ($("btnBc")) $("btnBc").addEventListener("click", onBc);
    if ($("btnCompare")) $("btnCompare").addEventListener("click", onCompare);
    if ($("onlyPrefer"))
      $("onlyPrefer").addEventListener("change", function () {
        loadRows();
      });
    if ($("showFixedDays"))
      $("showFixedDays").addEventListener("change", function () {
        loadRows();
      });
    if ($("yearFilter"))
      $("yearFilter").addEventListener("change", async function () {
        if (resyncInFlight || reviewDirty) {
          setMsg("切换审阅年前先等待自动重算完成（或点定稿）…");
          if (resyncInFlight) {
            var n = 0;
            while (resyncInFlight && n < 80) {
              await new Promise(function (r) {
                setTimeout(r, 200);
              });
              n += 1;
            }
          } else if (reviewDirty) {
            await doResync({ auto: true, force: true });
          }
        }
        loadRows();
      });
    if ($("evalYearFilter"))
      $("evalYearFilter").addEventListener("change", function () {
        renderEvalView();
      });
    if ($("compareYearFilter"))
      $("compareYearFilter").addEventListener("change", function () {
        renderComparePanel();
      });
  }

  async function init() {
    bind();
    ensureTrainChart();
    ensureWaterCharts();
    try {
      var boot = await fetch("/api/boot").then(function (r) {
        return r.json();
      });
      ACTION_SPACE = boot.action_space || [];
      if (boot.reason_codes && boot.reason_codes.length) {
        REASON_CODE_OPTIONS = boot.reason_codes.map(function (o) {
          return { code: o.code, label: o.label };
        });
      }
      if (boot.year_split) {
        expectedReviewYears = boot.year_split.review_years || [];
        if ($("reviewYearsLabel"))
          $("reviewYearsLabel").textContent = expectedReviewYears.join(",");
        if ($("holdoutYearsLabel"))
          $("holdoutYearsLabel").textContent = (boot.year_split.holdout_years || []).join(",");
      }
      applyRun(boot.run || {});
      applySession(boot.session || {});
      await refreshRunSelects();
      if (boot.run && boot.run.run_id) {
        await refreshSummary();
      }
      if (boot.train_state && boot.train_state.busy) startPoll();
      var params = new URLSearchParams(window.location.search);
      var step = Number(params.get("step") || 0);
      if (step >= 1 && step <= 6) {
        setStepper(step);
      } else {
        setStepper(inferStep());
      }
    } catch (e) {
      setMsg(String(e.message || e), true);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
