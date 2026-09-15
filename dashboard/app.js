/* Treasury Auction Phase 8 dashboard. Vanilla JS, no build step, no
 * external charting library -- reads the single generated JSON
 * aggregate at data/phase8_dashboard_data.json and renders everything
 * client-side. Cutoff and tenor are both global filters: every panel
 * except the "MAE by tenor" comparison chart is recomputed on the
 * selected tenor's own auctions (via the `*_by_tenor` tables the
 * generator precomputes), never left pooled while merely highlighting
 * a bar. The one exception is the by-tenor chart itself -- its whole
 * purpose is comparing all seven tenors side by side, so a tenor
 * selection there highlights that tenor among all of them instead of
 * hiding the other six; see its own chart note. No chart can ever mix
 * an announcement-cutoff number with a pre-auction one.
 */

(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const TENOR_ORDER = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"];

  const POINT_COLORS = {
    baseline_recent_history_adaptive: "var(--series-adaptive)",
    baseline_recent_history_frozen: "var(--series-frozen)",
    challenger_gbm_core: "var(--series-gbm)",
    challenger_shrinkage_tenor_reopening: "var(--series-shrinkage)",
  };
  const PROB_COLORS = {
    quantile_recent_history_residual: "var(--series-quantile-recent)",
    quantile_gbm_core: "var(--series-quantile-gbm)",
  };

  const state = {
    data: null,
    cutoff: "announcement",
    tenor: "__all__",
    showProvisional: false,
    interval: "90%",
  };

  function resolveVar(cssVarExpr) {
    // e.g. "var(--series-adaptive)" -> actual computed hex, so SVG
    // fill/stroke attributes (which don't accept CSS custom
    // properties directly in every context we use them) always work.
    const m = /var\((--[a-zA-Z0-9-]+)\)/.exec(cssVarExpr);
    if (!m) return cssVarExpr;
    return getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim();
  }

  function el(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function clear(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function sortTenors(tenors) {
    return tenors.slice().sort((a, b) => TENOR_ORDER.indexOf(a) - TENOR_ORDER.indexOf(b));
  }

  function setScopeNote(elementId, n) {
    const el2 = document.getElementById(elementId);
    if (!el2) return;
    el2.textContent =
      state.tenor === "__all__"
        ? "Scope: all tenors pooled" + (n !== null && n !== undefined ? ", n=" + n : "") + "."
        : "Scope: " + state.tenor + " only" + (n !== null && n !== undefined ? ", n=" + n : "") + " -- not pooled across tenors.";
  }

  // Point/probabilistic tables that are tenor-scoped when a tenor is
  // selected, else the pooled table -- the single switch every
  // tenor-sensitive panel below reads through, so none of them can
  // silently fall back to a pooled number while claiming to be filtered.
  function pointPooledRows() {
    return state.tenor === "__all__" ? state.data.point.pooled : state.data.point.pooled_by_tenor[state.tenor] || [];
  }
  function pointByYearRows() {
    return state.tenor === "__all__" ? state.data.point.by_year : state.data.point.by_year_by_tenor[state.tenor] || [];
  }
  function probabilisticByYearRows() {
    return state.tenor === "__all__"
      ? state.data.probabilistic.by_year
      : state.data.probabilistic.by_year_by_tenor[state.tenor] || [];
  }
  function probabilisticPooledRows() {
    return state.tenor === "__all__"
      ? state.data.probabilistic.pooled
      : state.data.probabilistic.pooled_by_tenor[state.tenor] || [];
  }
  function stressPooledForCutoff(cv) {
    return state.tenor === "__all__"
      ? state.data.stress.pooled[cv] || {}
      : (state.data.stress.pooled_by_tenor[state.tenor] || {})[cv] || {};
  }
  function caseStudyRows(cv) {
    return state.tenor === "__all__"
      ? state.data.point.case_studies[cv] || []
      : (state.data.point.case_studies_by_tenor[state.tenor] || {})[cv] || [];
  }

  // ============================================================
  // Data loading
  // ============================================================

  fetch("data/phase8_dashboard_data.json")
    .then((r) => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then((data) => {
      // Explicit success state: never rely on the load-error banner's
      // initial `hidden` HTML attribute alone -- set it here too, so a
      // successful load is guaranteed to hide it even if some earlier
      // code path (or a future edit) had already revealed it.
      document.getElementById("load-error").hidden = true;
      state.data = data;
      init(data);
    })
    .catch((err) => {
      // Explicit failure state: show the error banner and keep the main
      // dashboard hidden -- both set here, not left to the initial HTML.
      console.error("Failed to load dashboard data:", err);
      document.getElementById("load-error").hidden = false;
      document.getElementById("main").hidden = true;
    });

  function init(data) {
    document.getElementById("main").hidden = false;

    // Populate tenor filter from the by-tenor table's own tenor values.
    const tenorSelect = document.getElementById("tenor-select");
    const tenors = sortTenors(
      Array.from(new Set(data.point.by_tenor.map((r) => r.tenor)))
    );
    tenors.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      tenorSelect.appendChild(opt);
    });

    document.getElementById("cutoff-select").addEventListener("change", (e) => {
      state.cutoff = e.target.value;
      renderAll();
    });
    document.getElementById("tenor-select").addEventListener("change", (e) => {
      state.tenor = e.target.value;
      renderAll();
    });
    document.getElementById("provisional-toggle").addEventListener("change", (e) => {
      state.showProvisional = e.target.checked;
      renderAll();
    });
    document.getElementById("interval-select").addEventListener("change", (e) => {
      state.interval = e.target.value;
      renderAll();
    });

    document.getElementById("footer-digest").textContent =
      "Point-predictions digest: " + data.generated_from.point_predictions_digest +
      " | Probabilistic-predictions digest: " + data.generated_from.probabilistic_predictions_digest +
      " | " + data.generated_from.source;

    renderAll();
  }

  function renderAll() {
    renderKpis();
    renderMaeByYear();
    renderMaeByTenor();
    renderCoverageAndWidth();
    renderStress();
    renderCaseStudies();
  }

  // ============================================================
  // KPI tiles
  // ============================================================

  function renderKpis() {
    const cv = state.cutoff;
    const scopeLabel = state.tenor === "__all__" ? "complete years 2015–2025" : state.tenor + " only, complete years";
    const pooled = pointPooledRows().filter((r) => r.cutoff_view === cv);
    const frozen = pooled.find((r) => r.model_id === "baseline_recent_history_frozen");
    const adaptive = pooled.find((r) => r.model_id === "baseline_recent_history_adaptive");
    const probPooled = probabilisticPooledRows().find(
      (r) => r.cutoff_view === cv && r.model_id === "quantile_recent_history_residual" && r.scope === "complete_years"
    );
    const stress = stressPooledForCutoff(cv);

    const tiles = [
      {
        label: "Adaptive recent-history MAE",
        value: adaptive ? fmt(adaptive.mae, 2) : "n/a",
        unit: "pp, " + scopeLabel + ", n=" + (adaptive ? adaptive.n : "n/a"),
      },
      {
        label: "Frozen recent-history MAE",
        value: frozen ? fmt(frozen.mae, 2) : "n/a",
        unit: "pp, " + scopeLabel + ", n=" + (frozen ? frozen.n : "n/a"),
      },
      {
        label: "90% interval coverage",
        value: probPooled ? fmt(probPooled["coverage_90%"] * 100, 1) + "%" : "n/a",
        unit: "residual-quantile method, " + scopeLabel + " (nominal 90%)" + (probPooled ? ", n=" + probPooled.n : ""),
      },
      {
        label: "Stress classifier Brier vs. no-skill",
        value: stress.brier_score !== undefined && stress.brier_score !== null ? fmt(stress.brier_score, 3) + " vs " + fmt(stress.no_skill_brier_score, 3) : "n/a",
        unit:
          stress.n_positive !== undefined && stress.n_positive < 5
            ? "too few positive-labeled auctions (n_positive=" + (stress.n_positive || 0) + ") to read"
            : stress.worse_than_no_skill
              ? "worse than no-skill — poorly calibrated (n_positive=" + stress.n_positive + ")"
              : "n/a",
      },
    ];

    const grid = document.getElementById("kpi-grid");
    grid.innerHTML = "";
    tiles.forEach((t) => {
      const div = document.createElement("div");
      div.className = "kpi-tile";
      div.innerHTML =
        '<div class="kpi-label">' + t.label + "</div>" +
        '<div class="kpi-value">' + t.value + "</div>" +
        '<div class="kpi-unit">' + t.unit + "</div>";
      grid.appendChild(div);
    });
    setScopeNote("scope-note-kpi", adaptive ? adaptive.n : null);
  }

  // ============================================================
  // Chart: MAE by year (grouped bar, provisional 2026 hatched)
  // ============================================================

  function hatchPatternIdForColor(svg, colorHex) {
    // One pattern per distinct series color, defined with an EXPLICIT
    // fill (not `currentColor`) -- a shared `currentColor` pattern
    // would resolve against whatever `color` is inherited at the
    // <defs>/<pattern> location, NOT the bar referencing it, so every
    // provisional bar would render in one indistinguishable shade
    // regardless of its own series color. Explicit-per-color patterns
    // avoid that entirely.
    const safeId = "hatch-" + colorHex.replace(/[^a-zA-Z0-9]/g, "");
    let defs = svg.querySelector("defs");
    if (!defs) defs = el("defs", {}, svg);
    if (svg.querySelector("#" + safeId)) return safeId;
    const pattern = el(
      "pattern",
      { id: safeId, width: 6, height: 6, patternTransform: "rotate(45)", patternUnits: "userSpaceOnUse" },
      defs
    );
    el("rect", { width: 6, height: 6, fill: colorHex, "fill-opacity": 0.28 }, pattern);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: colorHex, "stroke-width": 2 }, pattern);
    return safeId;
  }

  function groupedBarChart(svg, opts) {
    // opts: {categories, series:[{key,label,color}], getValue(key,cat),
    //        isProvisional(cat), highlightCategory, unit, yLabel}
    clear(svg);
    const W = 960, H = 320;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMinYMin meet");

    const marginLeft = 46, marginRight = 12, marginTop = 16, marginBottom = 54;
    const plotW = W - marginLeft - marginRight;
    const plotH = H - marginTop - marginBottom;

    let maxVal = 0;
    opts.categories.forEach((cat) => {
      opts.series.forEach((s) => {
        const v = opts.getValue(s.key, cat);
        if (v !== null && v !== undefined && v > maxVal) maxVal = v;
      });
    });
    if (maxVal === 0) maxVal = 1;
    const niceMax = Math.ceil(maxVal * 1.15 / 5) * 5 || maxVal * 1.15;

    // gridlines + y-axis labels
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const y = marginTop + plotH - (i / ticks) * plotH;
      el("line", { class: "grid-line", x1: marginLeft, x2: W - marginRight, y1: y, y2: y }, svg);
      const label = el("text", { class: "axis-text", x: marginLeft - 8, y: y + 3, "text-anchor": "end" }, svg);
      label.textContent = fmt((niceMax * i) / ticks, 1);
    }
    el("line", { class: "axis-line", x1: marginLeft, x2: marginLeft, y1: marginTop, y2: marginTop + plotH }, svg);
    el("line", { class: "axis-line", x1: marginLeft, x2: W - marginRight, y1: marginTop + plotH, y2: marginTop + plotH }, svg);

    const groupW = plotW / opts.categories.length;
    const barPad = 6;
    const barW = (groupW - barPad * 2) / opts.series.length;

    opts.categories.forEach((cat, ci) => {
      const groupX = marginLeft + ci * groupW;
      const provisional = opts.isProvisional ? opts.isProvisional(cat) : false;

      if (opts.highlightCategory && cat === opts.highlightCategory) {
        el(
          "rect",
          {
            x: groupX + 1,
            y: marginTop,
            width: Math.max(groupW - 2, 1),
            height: plotH,
            fill: resolveVar("var(--focus-ring)"),
            "fill-opacity": 0.12,
            rx: 4,
          },
          svg
        );
      }

      opts.series.forEach((s, si) => {
        const v = opts.getValue(s.key, cat);
        const barX = groupX + barPad + si * barW;
        const barColor = resolveVar(s.color);
        if (v === null || v === undefined) return;
        const barH = (v / niceMax) * plotH;
        const barY = marginTop + plotH - barH;
        const rect = el(
          "rect",
          {
            class: "bar-rect",
            x: barX,
            y: barY,
            width: Math.max(barW - 2, 1),
            height: Math.max(barH, 0),
            fill: provisional ? "url(#" + hatchPatternIdForColor(svg, barColor) + ")" : barColor,
            stroke: provisional ? barColor : "none",
            "stroke-width": provisional ? 1.5 : 0,
            rx: 2,
            tabindex: 0,
          },
          svg
        );
        const n = opts.getN ? opts.getN(s.key, cat) : null;
        const title = el("title", {}, rect);
        title.textContent =
          s.label + " — " + cat + (provisional ? " (provisional)" : "") + ": " +
          fmt(v, 2) + " " + (opts.unit || "") + (n !== null ? " (n=" + n + ")" : "");
      });

      const catLabel = el(
        "text",
        { class: "axis-text", x: groupX + groupW / 2, y: marginTop + plotH + 18, "text-anchor": "middle" },
        svg
      );
      catLabel.textContent = cat + (provisional ? " (prov.)" : "");
    });

    if (opts.yLabel) {
      const yl = el(
        "text",
        {
          class: "axis-text",
          x: 14,
          y: marginTop + plotH / 2,
          "text-anchor": "middle",
          transform: "rotate(-90 14 " + (marginTop + plotH / 2) + ")",
        },
        svg
      );
      yl.textContent = opts.yLabel;
    }
  }

  function renderLegend(containerId, items) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    items.forEach((item) => {
      const div = document.createElement("div");
      div.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch" + (item.dashed ? " dashed" : "");
      swatch.style.background = item.dashed ? "none" : resolveVar(item.color);
      swatch.style.borderBottom = item.dashed ? "2px dashed " + resolveVar(item.color) : "none";
      const label = document.createElement("span");
      label.textContent = item.label;
      div.appendChild(swatch);
      div.appendChild(label);
      container.appendChild(div);
    });
  }

  function renderMaeByYear() {
    const data = state.data;
    const cv = state.cutoff;
    const rows = pointByYearRows().filter((r) => r.cutoff_view === cv);
    const provisionalYear = data.meta.provisional_year;
    let years = Array.from(new Set(rows.map((r) => r.test_year))).sort((a, b) => a - b);
    if (!state.showProvisional) years = years.filter((y) => y !== provisionalYear);

    const seriesDefs = data.meta.point_model_ids.map((id) => ({
      key: id,
      label: data.meta.point_model_labels[id] || id,
      color: POINT_COLORS[id] || "var(--series-adaptive)",
    }));

    const byYearModel = {};
    rows.forEach((r) => {
      byYearModel[r.test_year + "|" + r.model_id] = r;
    });

    groupedBarChart(document.getElementById("chart-mae-year"), {
      categories: years.map(String),
      series: seriesDefs,
      unit: "pp",
      yLabel: "MAE (pp)",
      isProvisional: (cat) => Number(cat) === provisionalYear,
      getValue: (key, cat) => {
        const row = byYearModel[cat + "|" + key];
        return row ? row.mae : null;
      },
      getN: (key, cat) => {
        const row = byYearModel[cat + "|" + key];
        return row ? row.n : null;
      },
    });

    renderLegend(
      "legend-mae-year",
      seriesDefs.map((s) => ({ label: s.label, color: s.color }))
    );
    const totalN = years.reduce((sum, y) => {
      const row = byYearModel[y + "|baseline_recent_history_adaptive"];
      return sum + (row ? row.n : 0);
    }, 0);
    setScopeNote("scope-note-mae-year", totalN || null);
  }

  // ============================================================
  // Chart: MAE by tenor
  // ============================================================

  function renderMaeByTenor() {
    const data = state.data;
    const cv = state.cutoff;
    const rows = data.point.by_tenor.filter((r) => r.cutoff_view === cv);
    const tenors = sortTenors(Array.from(new Set(rows.map((r) => r.tenor))));

    const seriesDefs = data.meta.point_model_ids.map((id) => ({
      key: id,
      label: data.meta.point_model_labels[id] || id,
      color: POINT_COLORS[id] || "var(--series-adaptive)",
    }));

    const byTenorModel = {};
    rows.forEach((r) => {
      byTenorModel[r.tenor + "|" + r.model_id] = r;
    });

    groupedBarChart(document.getElementById("chart-mae-tenor"), {
      categories: tenors,
      series: seriesDefs,
      unit: "pp",
      yLabel: "MAE (pp)",
      highlightCategory: state.tenor !== "__all__" ? state.tenor : null,
      getValue: (key, cat) => {
        const row = byTenorModel[cat + "|" + key];
        return row ? row.mae : null;
      },
      getN: (key, cat) => {
        const row = byTenorModel[cat + "|" + key];
        return row ? row.n : null;
      },
    });

    renderLegend(
      "legend-mae-tenor",
      seriesDefs.map((s) => ({ label: s.label, color: s.color }))
    );

    renderTenorDetail();
  }

  function renderTenorDetail() {
    const container = document.getElementById("tenor-detail");
    const data = state.data;
    const cv = state.cutoff;
    if (state.tenor === "__all__") {
      container.hidden = true;
      container.innerHTML = "";
      return;
    }
    container.hidden = false;
    const tenor = state.tenor;

    const maeRow = (modelId) =>
      data.point.by_tenor.find((r) => r.cutoff_view === cv && r.tenor === tenor && r.model_id === modelId);
    const paired = (data.point.paired_by_tenor[cv] || []).find((r) => r.tenor === tenor);
    const prob = (modelId) =>
      data.probabilistic.by_tenor.find((r) => r.cutoff_view === cv && r.tenor === tenor && r.model_id === modelId);

    const frozen = maeRow("baseline_recent_history_frozen");
    const adaptive = maeRow("baseline_recent_history_adaptive");
    const probRecent = prob("quantile_recent_history_residual");
    const interval = state.interval;

    let html = "<h3>" + tenor + " detail (" + (cv === "announcement" ? "announcement" : "pre-auction") + " cutoff, complete years)</h3><ul>";
    html += "<li>Frozen MAE: " + (frozen ? fmt(frozen.mae, 2) + " pp (n=" + frozen.n + ")" : "n/a") + "</li>";
    html += "<li>Adaptive MAE: " + (adaptive ? fmt(adaptive.mae, 2) + " pp (n=" + adaptive.n + ")" : "n/a") + "</li>";
    if (paired) {
      html +=
        "<li>Adaptive beat frozen on " + paired.n_model_better + " of " + paired.n +
        " auctions (frozen better on " + paired.n_benchmark_better + ", tied " + paired.n_tied + ")</li>";
    }
    if (probRecent) {
      html +=
        "<li>Residual-quantile " + interval + " coverage: " + fmt(probRecent["coverage_" + interval] * 100, 1) +
        "% (width " + fmt(probRecent["width_" + interval], 1) + " pp, n=" + probRecent.n + ")</li>";
    }
    html += "</ul>";
    container.innerHTML = html;
  }

  // ============================================================
  // Chart: coverage (line, with nominal dashed reference) + width (bar)
  // ============================================================

  function lineChart(svg, opts) {
    // opts: {categories, series:[{key,label,color}], getValue(key,cat), refValue, refLabel, yMax, yLabel, valueFmt}
    clear(svg);
    const W = 960, H = 300;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMinYMin meet");
    const marginLeft = 46, marginRight = 12, marginTop = 16, marginBottom = 40;
    const plotW = W - marginLeft - marginRight;
    const plotH = H - marginTop - marginBottom;
    const yMax = opts.yMax || 1;

    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const y = marginTop + plotH - (i / ticks) * plotH;
      el("line", { class: "grid-line", x1: marginLeft, x2: W - marginRight, y1: y, y2: y }, svg);
      const label = el("text", { class: "axis-text", x: marginLeft - 8, y: y + 3, "text-anchor": "end" }, svg);
      label.textContent = fmt((yMax * i) / ticks, 2);
    }
    el("line", { class: "axis-line", x1: marginLeft, x2: marginLeft, y1: marginTop, y2: marginTop + plotH }, svg);
    el("line", { class: "axis-line", x1: marginLeft, x2: W - marginRight, y1: marginTop + plotH, y2: marginTop + plotH }, svg);

    if (opts.refValue !== undefined && opts.refValue !== null) {
      const refY = marginTop + plotH - (opts.refValue / yMax) * plotH;
      el("line", { class: "ref-line", x1: marginLeft, x2: W - marginRight, y1: refY, y2: refY }, svg);
      const label = el("text", { class: "axis-text", x: W - marginRight, y: refY - 4, "text-anchor": "end" }, svg);
      label.textContent = opts.refLabel || "nominal";
    }

    const n = opts.categories.length;
    const step = n > 1 ? plotW / (n - 1) : 0;

    opts.categories.forEach((cat, i) => {
      const x = marginLeft + (n > 1 ? i * step : plotW / 2);
      const label = el("text", { class: "axis-text", x: x, y: marginTop + plotH + 18, "text-anchor": "middle" }, svg);
      label.textContent = cat;
    });

    opts.series.forEach((s) => {
      const color = resolveVar(s.color);
      const points = [];
      opts.categories.forEach((cat, i) => {
        const v = opts.getValue(s.key, cat);
        if (v === null || v === undefined) return;
        const x = marginLeft + (n > 1 ? i * step : plotW / 2);
        const y = marginTop + plotH - (v / yMax) * plotH;
        points.push([x, y, v]);
      });
      if (points.length > 1) {
        const d = points.map((p, i) => (i === 0 ? "M" : "L") + p[0] + " " + p[1]).join(" ");
        el("path", { class: "data-line", d: d, stroke: color }, svg);
      }
      points.forEach((p) => {
        const dot = el("circle", { class: "data-dot", cx: p[0], cy: p[1], r: 3.5, fill: color }, svg);
        const title = el("title", {}, dot);
        title.textContent = s.label + ": " + (opts.valueFmt ? opts.valueFmt(p[2]) : fmt(p[2], 3));
      });
    });

    if (opts.yLabel) {
      const yl = el(
        "text",
        {
          class: "axis-text",
          x: 14,
          y: marginTop + plotH / 2,
          "text-anchor": "middle",
          transform: "rotate(-90 14 " + (marginTop + plotH / 2) + ")",
        },
        svg
      );
      yl.textContent = opts.yLabel;
    }
  }

  function renderCoverageAndWidth() {
    const data = state.data;
    const cv = state.cutoff;
    const interval = state.interval;
    const covCol = "coverage_" + interval;
    const widthCol = "width_" + interval;
    const nominalFraction = { "50%": 0.5, "80%": 0.8, "90%": 0.9 }[interval];

    const rows = probabilisticByYearRows().filter((r) => r.cutoff_view === cv);
    const years = Array.from(new Set(rows.map((r) => r.test_year))).sort((a, b) => a - b);
    const byYearModel = {};
    rows.forEach((r) => {
      byYearModel[r.test_year + "|" + r.model_id] = r;
    });

    const seriesDefs = data.meta.probabilistic_model_ids.map((id) => ({
      key: id,
      label: data.meta.probabilistic_model_labels[id] || id,
      color: PROB_COLORS[id] || "var(--series-quantile-recent)",
    }));

    lineChart(document.getElementById("chart-coverage-year"), {
      categories: years.map(String),
      series: seriesDefs,
      getValue: (key, cat) => {
        const row = byYearModel[cat + "|" + key];
        return row ? row[covCol] : null;
      },
      refValue: nominalFraction,
      refLabel: "nominal " + interval,
      yMax: 1,
      yLabel: "Empirical coverage",
      valueFmt: (v) => fmt(v * 100, 1) + "%",
    });

    groupedBarChart(document.getElementById("chart-width-year"), {
      categories: years.map(String),
      series: seriesDefs,
      unit: "pp",
      yLabel: "Mean width (pp)",
      getValue: (key, cat) => {
        const row = byYearModel[cat + "|" + key];
        return row ? row[widthCol] : null;
      },
      getN: (key, cat) => {
        const row = byYearModel[cat + "|" + key];
        return row ? row.n : null;
      },
    });

    renderLegend(
      "legend-coverage",
      seriesDefs
        .map((s) => ({ label: s.label, color: s.color }))
        .concat([{ label: "Nominal " + interval + " (dashed)", color: "var(--text-muted)", dashed: true }])
    );
    const totalN = years.reduce((sum, y) => {
      const row = byYearModel[y + "|quantile_recent_history_residual"];
      return sum + (row ? row.n : 0);
    }, 0);
    setScopeNote("scope-note-coverage", totalN || null);
  }

  // ============================================================
  // Chart: stress classifier vs no-skill / PR-AUC vs base rate
  // ============================================================

  function simpleBarChart(svg, opts) {
    // opts: {categories, values, colors, unit, yLabel}
    clear(svg);
    const W = 440, H = 300;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMinYMin meet");
    const marginLeft = 46, marginRight = 12, marginTop = 16, marginBottom = 40;
    const plotW = W - marginLeft - marginRight;
    const plotH = H - marginTop - marginBottom;
    const maxVal = Math.max.apply(null, opts.values.filter((v) => v !== null && v !== undefined).concat([0.0001]));
    const niceMax = maxVal * 1.25;

    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const y = marginTop + plotH - (i / ticks) * plotH;
      el("line", { class: "grid-line", x1: marginLeft, x2: W - marginRight, y1: y, y2: y }, svg);
      const label = el("text", { class: "axis-text", x: marginLeft - 8, y: y + 3, "text-anchor": "end" }, svg);
      label.textContent = fmt((niceMax * i) / ticks, 3);
    }
    el("line", { class: "axis-line", x1: marginLeft, x2: marginLeft, y1: marginTop, y2: marginTop + plotH }, svg);
    el("line", { class: "axis-line", x1: marginLeft, x2: W - marginRight, y1: marginTop + plotH, y2: marginTop + plotH }, svg);

    const slotW = plotW / opts.categories.length;
    opts.categories.forEach((cat, i) => {
      const v = opts.values[i];
      const barW = slotW * 0.5;
      const x = marginLeft + i * slotW + (slotW - barW) / 2;
      const barH = v === null ? 0 : (v / niceMax) * plotH;
      const y = marginTop + plotH - barH;
      const rect = el(
        "rect",
        { class: "bar-rect", x: x, y: y, width: barW, height: Math.max(barH, 0), fill: resolveVar(opts.colors[i]), rx: 3, tabindex: 0 },
        svg
      );
      const title = el("title", {}, rect);
      title.textContent = cat + ": " + fmt(v, 4);
      const valLabel = el("text", { class: "value-label", x: x + barW / 2, y: y - 6, "text-anchor": "middle" }, svg);
      valLabel.textContent = fmt(v, 3);
      const catLabel = el("text", { class: "axis-text", x: marginLeft + i * slotW + slotW / 2, y: marginTop + plotH + 18, "text-anchor": "middle" }, svg);
      catLabel.textContent = cat;
    });

    if (opts.yLabel) {
      const yl = el(
        "text",
        {
          class: "axis-text",
          x: 14,
          y: marginTop + plotH / 2,
          "text-anchor": "middle",
          transform: "rotate(-90 14 " + (marginTop + plotH / 2) + ")",
        },
        svg
      );
      yl.textContent = opts.yLabel;
    }
  }

  const STRESS_MIN_POSITIVE = 5; // same per-group data-sufficiency gate documented in the chart note and phase_8_interpretation.md

  function renderStress() {
    const cv = state.cutoff;
    const stress = stressPooledForCutoff(cv);
    const nPositive = stress.n_positive || 0;
    const insufficientBanner = document.getElementById("stress-tenor-insufficient");

    if (state.tenor !== "__all__" && nPositive < STRESS_MIN_POSITIVE) {
      insufficientBanner.hidden = false;
      insufficientBanner.innerHTML =
        '<span aria-hidden="true">&#9888;</span><span>Only ' + nPositive + " positive-labeled auction" +
        (nPositive === 1 ? "" : "s") + " for " + state.tenor +
        " (n=" + (stress.n || 0) + " total) -- below this project's own &ge;5-per-group gate. " +
        "The numbers below are shown for transparency but should not be read as a reliable estimate at this sample size.</span>";
    } else {
      insufficientBanner.hidden = true;
      insufficientBanner.innerHTML = "";
    }

    simpleBarChart(document.getElementById("chart-stress-brier"), {
      categories: ["Classifier", "No-skill"],
      values: [stress.brier_score, stress.no_skill_brier_score],
      colors: ["var(--status-critical)", "var(--series-no-skill)"],
      yLabel: "Brier score (lower is better)",
    });
    const baseRate = stress.prevalence;
    simpleBarChart(document.getElementById("chart-stress-prauc"), {
      categories: ["PR-AUC", "Base rate"],
      values: [stress.pr_auc, baseRate],
      colors: ["var(--series-gbm)", "var(--series-no-skill)"],
      yLabel: "PR-AUC / base rate",
    });

    renderLegend("legend-stress", [
      { label: "Classifier (n=" + (stress.n || "n/a") + ", " + nPositive + " positive)", color: "var(--status-critical)" },
      { label: "No-skill / base rate reference", color: "var(--series-no-skill)" },
    ]);
    setScopeNote("scope-note-stress", stress.n || null);
  }

  // ============================================================
  // Case studies table
  // ============================================================

  function renderCaseStudies() {
    const cv = state.cutoff;
    const rows = caseStudyRows(cv);
    const table = document.getElementById("case-studies-table");
    if (!rows.length) {
      table.innerHTML =
        "<tr><td>No case studies available for this " +
        (state.tenor === "__all__" ? "cutoff." : "tenor/cutoff combination.") +
        "</td></tr>";
      setScopeNote("scope-note-case-studies", 0);
      return;
    }
    const cols = [
      ["case_selection_rule", "Selection rule"],
      ["auction_key", "Auction"],
      ["test_year", "Year"],
      ["tenor", "Tenor"],
      ["is_reopening", "Reopening?"],
      ["known_at_cutoff__frozen_forecast_pct", "Frozen forecast (%)"],
      ["known_at_cutoff__adaptive_forecast_pct", "Adaptive forecast (%)"],
      ["learned_after_settlement__actual_pct", "Actual (settled, %)"],
      ["frozen_abs_error_pp", "Frozen |error| (pp)"],
      ["adaptive_abs_error_pp", "Adaptive |error| (pp)"],
      ["adaptive_improvement_pp", "Adaptive improvement (pp)"],
    ];
    let html = "<thead><tr>" + cols.map((c) => "<th>" + c[1] + "</th>").join("") + "</tr></thead><tbody>";
    rows.forEach((row) => {
      html += "<tr>" + cols.map((c) => {
        let v = row[c[0]];
        if (typeof v === "number") v = fmt(v, 2);
        if (typeof v === "boolean") v = v ? "Yes" : "No";
        return "<td>" + v + "</td>";
      }).join("") + "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
    setScopeNote("scope-note-case-studies", rows.length);
  }
})();
