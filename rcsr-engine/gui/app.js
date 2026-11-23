"use strict";

// The front end consumes only the schema, the NDJSON event stream and the
// bundled example. It never scrapes human-readable output.

const ROLE_FLAGS = new Set(["--data", "--target", "--vars", "--time", "--group-by", "--weight"]);
const PINNED = ["--metric", "--generations", "--pop", "--islands", "--max-complexity", "--seed"];
const ROLE_NAMES = [
  ["input", "input"],
  ["target", "target"],
  ["time", "time"],
  ["group", "group"],
  ["weight", "weight"],
  ["ignore", "ignore"],
];

const state = {
  schema: null,
  headers: [],
  rows: [],
  numeric: [],
  roles: [],
  tableText: "",
  fileName: "",
  fields: [],
  solutions: [],
  selected: -1,
  running: false,
  start: null,
  baselines: null,
  residual: null,
  cv: null,
  stability: null,
  nullControl: null,
  done: null,
  log: [],
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmt(v) {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return String(Number(v.toPrecision(6)));
}

function pct(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return (100 * v).toFixed(1) + "%";
}

function shellQuote(s) {
  if (/^[A-Za-z0-9_./:=+-]+$/.test(s)) return s;
  return "'" + s.replace(/'/g, "'\\''") + "'";
}

function setMessage(id, text, kind) {
  const node = $(id);
  node.textContent = text || "";
  node.className = "message" + (kind ? " " + kind : "");
}

function setStatus(text, kind) {
  const badge = $("statusBadge");
  badge.textContent = text;
  badge.className = "pill " + (kind || "ready");
  $("footerStatus").textContent = text;
}

// ---------------------------------------------------------------------------
// Table parsing (client-side preview only)
// ---------------------------------------------------------------------------

function splitDelimited(line, delim) {
  const out = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += c;
      }
    } else if (c === '"' && field === "") {
      quoted = true;
    } else if (c === delim) {
      out.push(field);
      field = "";
    } else {
      field += c;
    }
  }
  out.push(field);
  return out;
}

function parseTable(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) throw new Error("the table is empty");
  const headerLine = lines[0];
  const delim = headerLine.includes("\t") ? "\t" : headerLine.includes(",") ? "," : null;
  const split = (line) => (delim ? splitDelimited(line, delim) : line.trim().split(/\s+/));
  const headers = split(headerLine).map((s) => s.trim());
  if (headers.some((h) => h === "")) throw new Error("a column name is empty");
  const rows = lines.slice(1).map((line) => {
    const cells = split(line).map((s) => s.trim());
    while (cells.length < headers.length) cells.push("");
    return cells.slice(0, headers.length);
  });
  if (rows.length === 0) throw new Error("the table has no data rows");
  return { headers, rows };
}

function isMissing(cell) {
  return cell === "" || /^(na|nan|null)$/i.test(cell.trim());
}

function columnNumeric(rows, j) {
  return rows.every((row) => isMissing(row[j]) || Number.isFinite(Number(row[j])));
}

function defaultRoles(headers, rows) {
  const numeric = headers.map((_, j) => columnNumeric(rows, j));
  const roles = headers.map((h, j) => {
    if (/^(time|date|timestamp|datetime)$/i.test(h) || /(^|_)(time|date|timestamp)$/i.test(h)) return "time";
    if (/(^|_)(group|subject|site|subject_id|symbol|ticker|panel)$/i.test(h)) return "group";
    return numeric[j] ? "input" : "ignore";
  });
  let target = headers.findIndex((h) => /^(y|target|label|response|outcome)$/i.test(h));
  if (target < 0) {
    for (let j = headers.length - 1; j >= 0; j -= 1) {
      if (numeric[j]) {
        target = j;
        break;
      }
    }
  }
  if (target >= 0) roles[target] = "target";
  // Only one time and one group role by default.
  let seenTime = false;
  let seenGroup = false;
  roles.forEach((r, j) => {
    if (r === "time") {
      if (seenTime) roles[j] = numeric[j] ? "input" : "ignore";
      seenTime = true;
    }
    if (r === "group") {
      if (seenGroup) roles[j] = numeric[j] ? "input" : "ignore";
      seenGroup = true;
    }
  });
  if (!roles.includes("input") && headers.length > 1) {
    const j = headers.findIndex((h, k) => roles[k] === "ignore" && numeric[k]);
    if (j >= 0) roles[j] = "input";
  }
  return { numeric, roles };
}

function applyTable(text) {
  const parsed = parseTable(text);
  const { numeric, roles } = defaultRoles(parsed.headers, parsed.rows);
  state.headers = parsed.headers;
  state.rows = parsed.rows;
  state.numeric = numeric;
  state.roles = roles;
  state.tableText = text;
  state.solutions = [];
  state.selected = -1;
  state.baselines = null;
  state.residual = null;
  state.cv = null;
  state.stability = null;
  state.nullControl = null;
  state.done = null;
  renderRoles();
  renderPreview();
  renderSolutions();
  renderDetail();
  updateCli();
  $("rowCount").textContent = parsed.rows.length + " rows × " + parsed.headers.length + " cols";
  setMessage("dataMessage", "parsed " + parsed.rows.length + " rows", "ok");
}

function renderRoles() {
  const table = $("grid");
  table.textContent = "";
  const head = el("tr");
  const roleHead = el("tr");
  state.headers.forEach((name, j) => {
    const th = el("th", null, name);
    if (!state.numeric[j]) th.title = "non-numeric column";
    head.appendChild(th);
    const td = el("td", "role");
    const select = el("select");
    select.setAttribute("aria-label", "role for " + name);
    ROLE_NAMES.forEach(([value, label]) => {
      const option = el("option", null, label);
      option.value = value;
      if (state.roles[j] === value) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      const value = select.value;
      if (value === "target" || value === "time" || value === "group" || value === "weight") {
        state.roles = state.roles.map((r, k) => {
          if (k === j) return value;
          if (r === value) return state.numeric[k] ? "input" : "ignore";
          return r;
        });
      } else {
        state.roles[j] = value;
      }
      renderRoles();
      updateCli();
    });
    td.appendChild(select);
    roleHead.appendChild(td);
  });
  table.appendChild(head);
  table.appendChild(roleHead);
}

function renderPreview() {
  const table = $("preview");
  table.textContent = "";
  if (state.headers.length === 0) {
    $("previewNote").textContent = "No table parsed.";
    return;
  }
  const head = el("tr");
  head.appendChild(el("th", null, "#"));
  state.headers.forEach((name, j) => {
    const th = el("th", null, name + " · " + state.roles[j]);
    head.appendChild(th);
  });
  table.appendChild(head);
  const limit = Math.min(state.rows.length, 200);
  for (let i = 0; i < limit; i += 1) {
    const tr = el("tr");
    tr.appendChild(el("td", "num", String(i + 1)));
    state.rows[i].forEach((cell, j) => {
      tr.appendChild(el("td", state.numeric[j] ? "num" : null, cell));
    });
    table.appendChild(tr);
  }
  $("previewNote").textContent =
    state.rows.length > limit
      ? "showing the first " + limit + " of " + state.rows.length + " rows"
      : state.rows.length + " rows";
}

// ---------------------------------------------------------------------------
// Schema-driven form
// ---------------------------------------------------------------------------

function enumChoices(option) {
  let text = option.description || "";
  const colon = text.lastIndexOf(":");
  if (colon >= 0) text = text.slice(colon + 1);
  const parts = text
    .split("|")
    .map((part) =>
      part
        .replace(/\([^)]*\)/g, "")
        .trim()
        .replace(/^(subtract|divide by|missing-value method:)\s+/i, "")
        .split(/\s+/)[0]
    )
    .filter((part) => part && /^[a-z0-9][a-z0-9_-]*$/i.test(part));
  if (option.default && !["none", ""].includes(option.default) && !parts.includes(option.default)) {
    parts.unshift(option.default);
  }
  return [...new Set(parts)];
}

let opsDatalist = null;

function ensureOpsDatalist() {
  if (opsDatalist) return opsDatalist;
  opsDatalist = el("datalist");
  opsDatalist.id = "opsDatalist";
  (state.schema.operators || []).forEach((op) => {
    const option = el("option");
    option.value = op;
    opsDatalist.appendChild(option);
  });
  document.body.appendChild(opsDatalist);
  return opsDatalist;
}

function makeField(option) {
  const wrap = el("div", "field" + (option.type === "flag" ? " check" : ""));
  const id = "opt" + option.name.replace(/^-+/, "").replace(/[^a-z0-9]/gi, "_");
  const label = el("label", null, option.name.replace(/^--/, ""));
  label.htmlFor = id;
  let control;
  if (option.type === "flag") {
    control = el("input");
    control.type = "checkbox";
    control.checked = false;
  } else if (option.type === "enum") {
    control = el("select");
    enumChoices(option).forEach((choice) => {
      const opt = el("option", null, choice);
      opt.value = choice;
      if (choice === option.default) opt.selected = true;
      control.appendChild(opt);
    });
    if (![...control.options].some((o) => o.value === option.default)) {
      const opt = el("option", null, option.default);
      opt.value = option.default;
      opt.selected = true;
      control.appendChild(opt);
    }
  } else {
    control = el("input");
    control.type = option.type === "int" || option.type === "float" ? "number" : "text";
    if (control.type === "number") {
      control.step = option.type === "int" ? "1" : "any";
      if (option.default !== "none" && option.default !== "" && Number.isFinite(Number(option.default))) {
        control.value = option.default;
      }
    } else {
      control.value = ["none", "all-1", "all-continuous", "builtin-set", "all-except-target"].includes(option.default)
        ? ""
        : option.default;
      if (option.name === "--ops") {
        ensureOpsDatalist();
        control.setAttribute("list", "opsDatalist");
      }
    }
  }
  control.id = id;
  control.dataset.flag = option.name;
  control.dataset.otype = option.type;
  control.dataset.default = option.default;
  control.addEventListener("input", updateCli);
  control.addEventListener("change", updateCli);
  if (option.type === "flag") {
    wrap.appendChild(control);
    wrap.appendChild(label);
  } else {
    wrap.appendChild(label);
    wrap.appendChild(control);
  }
  if (option.description) {
    const desc = el("div", "desc", option.description);
    wrap.appendChild(desc);
  }
  state.fields.push(control);
  return wrap;
}

function renderForm() {
  const options = state.schema.options || [];
  const pinned = $("pinned");
  const more = $("more");
  pinned.textContent = "";
  more.textContent = "";
  const byName = new Map(options.map((o) => [o.name, o]));
  PINNED.forEach((name) => {
    if (byName.has(name)) pinned.appendChild(makeField(byName.get(name)));
  });
  options.forEach((option) => {
    if (ROLE_FLAGS.has(option.name) || PINNED.includes(option.name)) return;
    more.appendChild(makeField(option));
  });
}

function formValues() {
  const values = [];
  state.fields.forEach((control) => {
    const flag = control.dataset.flag;
    const type = control.dataset.otype;
    const def = control.dataset.default;
    if (type === "flag") {
      if (control.checked) values.push(flag);
      return;
    }
    const value = control.value.trim();
    if (value === "") return;
    if (type === "enum" && value === def) return;
    if (type === "int" || type === "float") {
      if (Number.isFinite(Number(def)) && Number(value) === Number(def)) return;
    } else if (value === def) {
      return;
    }
    values.push(flag, value);
  });
  return values;
}

function buildArgs() {
  const args = [];
  const byRole = (role) => state.headers.filter((_, j) => state.roles[j] === role);
  const target = byRole("target")[0];
  if (target === undefined) throw new Error("choose a target column in the Data pane");
  const inputs = byRole("input");
  if (inputs.length === 0) throw new Error("choose at least one input column in the Data pane");
  args.push("--data", "-", "--target", target, "--vars", inputs.join(","));
  const time = byRole("time")[0];
  const group = byRole("group")[0];
  const weight = byRole("weight")[0];
  if (time !== undefined) args.push("--time", time);
  if (group !== undefined) args.push("--group-by", group);
  if (weight !== undefined) args.push("--weight", weight);
  args.push(...formValues());
  return args;
}

function commandText() {
  let args;
  try {
    args = buildArgs();
  } catch (e) {
    return "rcsr --data - --target …";
  }
  return ["rcsr", ...args.map(shellQuote)].join(" ");
}

function copyText() {
  let args;
  try {
    args = buildArgs();
  } catch (e) {
    return "";
  }
  const command = ["rcsr", ...args.map(shellQuote)].join(" ");
  if (!state.tableText.trim()) return command;
  return command + " <<'RCSR_DATA'\n" + state.tableText.replace(/\s+$/, "") + "\nRCSR_DATA\n";
}

function updateCli() {
  $("cliBar").textContent = commandText();
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

function resetRun() {
  state.solutions = [];
  state.selected = -1;
  state.start = null;
  state.baselines = null;
  state.residual = null;
  state.cv = null;
  state.stability = null;
  state.nullControl = null;
  state.done = null;
  state.log = [];
  $("runLog").textContent = "";
  renderSolutions();
  renderDetail();
  drawPareto();
}

function appendLog(text) {
  state.log.push(text);
  const log = $("runLog");
  log.textContent = state.log.join("\n");
  log.scrollTop = log.scrollHeight;
}

async function startRun() {
  if (state.running) return;
  if (!state.tableText.trim()) {
    setMessage("formMessage", "load or paste a table first", "error");
    return;
  }
  let args;
  try {
    args = buildArgs();
  } catch (e) {
    setMessage("formMessage", e.message, "error");
    return;
  }
  resetRun();
  state.running = true;
  $("runBtn").disabled = true;
  setStatus("running", "running");
  setMessage("formMessage", "");
  switchTab("run");
  $("progressLine").textContent = "starting…";
  appendLog("POST /api/run");
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ args, table: state.tableText }),
    });
    if (!response.ok) {
      let message = "HTTP " + response.status;
      try {
        const body = await response.json();
        if (body.error) message = body.error;
      } catch (e) {
        /* keep the status line */
      }
      throw new Error(message);
    }
    if (!response.body) throw new Error("this browser cannot stream the response");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, cut).trim();
        buffer = buffer.slice(cut + 1);
        if (line) consumeLine(line);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) consumeLine(buffer.trim());
  } catch (e) {
    appendLog("error: " + e.message);
    setMessage("formMessage", e.message, "error");
    setStatus("error", "error");
  } finally {
    state.running = false;
    $("runBtn").disabled = false;
  }
}

function consumeLine(line) {
  let event;
  try {
    event = JSON.parse(line);
  } catch (e) {
    appendLog("unparsable stream line: " + line);
    return;
  }
  switch (event.event) {
    case "start":
      state.start = event;
      appendLog(
        "start: " +
          event.split.mode +
          " split · " +
          event.train_rows +
          " train / " +
          event.test_rows +
          " test · metric " +
          event.metric
      );
      break;
    case "note":
      appendLog("note: " + event.message);
      break;
    case "preparation":
      appendLog("preparation: fill=" + event.fill + " center=" + event.center + " scale=" + event.scale);
      break;
    case "progress":
      $("progressLine").textContent =
        "generation " +
        event.generation +
        "/" +
        event.total +
        " · best " +
        fmt(event.best_error) +
        " · " +
        event.pareto_members +
        " on the front";
      $("footerMeta").textContent = "generation " + event.generation + "/" + event.total;
      break;
    case "solution":
      state.solutions.push(event);
      renderSolutions();
      drawPareto();
      if (state.selected < 0) selectSolution(0);
      break;
    case "baselines":
      state.baselines = event;
      drawPareto();
      renderDetail();
      break;
    case "residual_tests":
      state.residual = event;
      renderDetail();
      break;
    case "cross_validation":
      state.cv = event;
      renderDetail();
      break;
    case "stability":
      state.stability = event;
      renderDetail();
      break;
    case "null_control":
      state.nullControl = event;
      renderDetail();
      break;
    case "eval":
      appendLog("eval: " + event.expression + " · train " + fmt(event.train_error) + " · test " + fmt(event.test_error));
      break;
    case "done":
      state.done = event;
      $("progressLine").textContent =
        "done · " +
        event.generations_run +
        " generations · " +
        fmt(event.elapsed_seconds) +
        " s · " +
        event.pareto_members +
        " solutions";
      setStatus("done", "ready");
      appendLog("done: " + event.pareto_members + " Pareto member(s)");
      renderSolutions();
      drawPareto();
      break;
    case "error":
      appendLog("error: " + event.message);
      setMessage("formMessage", event.message, "error");
      setStatus("error", "error");
      break;
    default:
      appendLog("event: " + JSON.stringify(event));
  }
}

// ---------------------------------------------------------------------------
// Solutions and detail
// ---------------------------------------------------------------------------

function renderSolutions() {
  const body = $("solutionsTable").querySelector("tbody");
  body.textContent = "";
  state.solutions.forEach((solution, index) => {
    const tr = el("tr", "solution" + (index === state.selected ? " selected" : ""));
    tr.tabIndex = 0;
    tr.addEventListener("click", () => selectSolution(index));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectSolution(index);
      }
    });
    tr.appendChild(el("td", "num", solution.rank));
    tr.appendChild(el("td", "num", fmt(solution.complexity)));
    tr.appendChild(el("td", "num", solution.k_constants));
    tr.appendChild(el("td", "num", fmt(solution.train_error)));
    tr.appendChild(el("td", "num", fmt(solution.test_error)));
    tr.appendChild(el("td", "num", fmt(solution.aicc)));
    const expr = el("td", "expr", solution.expression);
    expr.title = solution.expression;
    tr.appendChild(expr);
    body.appendChild(tr);
  });
  $("solutionCount").textContent =
    state.solutions.length === 0 ? "no solutions" : state.solutions.length + " solutions";
}

function selectSolution(index) {
  state.selected = index;
  renderSolutions();
  renderDetail();
  drawPareto();
}

function definitionList(rows) {
  const dl = el("dl", "kv");
  rows.forEach(([key, value]) => {
    dl.appendChild(el("dt", null, key));
    const dd = el("dd");
    if (value instanceof Node) dd.appendChild(value);
    else dd.textContent = value;
    dl.appendChild(dd);
  });
  return dl;
}

function renderDetail() {
  const body = $("detailBody");
  const formula = $("detailFormula");
  body.textContent = "";
  const solution = state.solutions[state.selected];
  if (!solution) {
    formula.textContent = "Run a search, then pick a row.";
    return;
  }
  formula.textContent = "y = " + solution.expression;

  body.appendChild(el("h3", null, "Fit and complexity"));
  body.appendChild(
    definitionList([
      ["complexity", fmt(solution.complexity)],
      ["k_constants", String(solution.k_constants)],
      ["train error", fmt(solution.train_error)],
      ["test error", fmt(solution.test_error)],
      ["overfit index", fmt(solution.overfit_index)],
      ["AICc", fmt(solution.aicc)],
      ["BIC", fmt(solution.bic)],
      ["Durbin–Watson", fmt(solution.durbin_watson)],
    ])
  );
  if (Array.isArray(solution.fold_errors) && solution.fold_errors.length > 1) {
    body.appendChild(el("h3", null, "Per-block test error"));
    body.appendChild(definitionList(solution.fold_errors.map((value, i) => ["block " + (i + 1), fmt(value)])));
  }

  body.appendChild(el("h3", null, "Constants ± standard error"));
  const constants = solution.constant_errors;
  if (constants && Array.isArray(constants.constants) && constants.constants.length > 0) {
    body.appendChild(
      definitionList(
        constants.constants.map((c, i) => [
          "constant " + (i + 1),
          fmt(c.value) + " ± " + fmt(c.std_error) + " (" + pct(c.relative) + " relative)",
        ])
      )
    );
    if (constants.ill_conditioned) {
      body.appendChild(el("p", "message error", "The fit is ill-conditioned: the data does not pin these constants down separately."));
    }
  } else {
    body.appendChild(el("p", "muted small", "No fitted constants, or too few residual degrees of freedom."));
  }

  body.appendChild(el("h3", null, "Residual checks"));
  if (state.residual) {
    const r = state.residual;
    body.appendChild(
      definitionList([
        ["n", String(r.n)],
        ["Ljung–Box p", fmt(r.ljung_box.p_value) + " (df " + r.ljung_box.df + ", statistic " + fmt(r.ljung_box.statistic) + ")"],
        ["Breusch–Pagan p", fmt(r.breusch_pagan.p_value) + " (df " + r.breusch_pagan.df + ")"],
        ["skewness", fmt(r.skewness)],
        ["excess kurtosis", fmt(r.kurtosis)],
      ])
    );
    body.appendChild(el("p", "muted small", "Residual tests are computed for the best training-error front member."));
  } else {
    body.appendChild(el("p", "muted small", "Residual tests arrive after the front is complete."));
  }

  body.appendChild(el("h3", null, "Baselines (must beat them)"));
  if (state.baselines) {
    const b = state.baselines;
    body.appendChild(
      definitionList([
        ["constant mean", "train " + fmt(b.mean.train_error) + " · test " + fmt(b.mean.test_error)],
        ["least-squares line", "train " + fmt(b.linear.train_error) + " · test " + fmt(b.linear.test_error)],
        ["k-NN (k=" + b.knn.k + ")", "train " + fmt(b.knn.train_error) + " · test " + fmt(b.knn.test_error)],
        ["beats all three?", beatsBaselines(solution) ? "yes on test error" : "no on test error"],
      ])
    );
  } else {
    body.appendChild(el("p", "muted small", "Baselines arrive after the front is complete."));
  }

  body.appendChild(el("h3", null, "Bootstrap"));
  if (solution.bootstrap_error) {
    const b = solution.bootstrap_error;
    body.appendChild(
      definitionList([
        ["apparent", fmt(b.apparent)],
        ["out of bag", fmt(b.oob)],
        [".632+ estimate", fmt(b.estimate)],
        ["weight", fmt(b.weight)],
      ])
    );
  } else {
    body.appendChild(el("p", "muted small", "Not requested; add --bootstrap-error <n>."));
  }

  body.appendChild(el("h3", null, "Cross-validation"));
  if (state.cv) {
    body.appendChild(
      definitionList([
        ["mode", state.cv.mode],
        ["folds", String(state.cv.k)],
        ["mean train", fmt(state.cv.mean_train_error)],
        ["mean test", fmt(state.cv.mean_test_error)],
        ["spread", fmt(state.cv.std_test_error)],
      ])
    );
  } else {
    body.appendChild(el("p", "muted small", "Not requested; add --cv k-fold|rolling|purged."));
  }

  body.appendChild(el("h3", null, "Stability and null control"));
  if (solution.stability !== null && solution.stability !== undefined) {
    body.appendChild(definitionList([["form frequency", pct(solution.stability)]]));
  } else {
    body.appendChild(el("p", "muted small", "Stability not requested; add --stability <B>."));
  }
  if (state.nullControl) {
    body.appendChild(
      definitionList([
        ["null runs", String(state.nullControl.runs)],
        ["null minimum", fmt(state.nullControl.min)],
        ["p-value", fmt(state.nullControl.p_value)],
        ["verdict", state.nullControl.verdict],
      ])
    );
  } else {
    body.appendChild(el("p", "muted small", "No y-randomisation control; add --null-runs <n>."));
  }
}

function beatsBaselines(solution) {
  if (!state.baselines || !Number.isFinite(solution.test_error)) return false;
  return (
    solution.test_error < state.baselines.mean.test_error &&
    solution.test_error < state.baselines.linear.test_error &&
    solution.test_error < state.baselines.knn.test_error
  );
}

// ---------------------------------------------------------------------------
// Pareto chart
// ---------------------------------------------------------------------------

function drawPareto() {
  const canvas = $("pareto");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width * dpr));
  const height = Math.max(240, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = width / dpr;
  const H = height / dpr;
  const pad = { left: 66, right: 18, top: 18, bottom: 42 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const styles = getComputedStyle(document.documentElement);
  const ink = styles.getPropertyValue("--ink").trim() || "#1f2933";
  const muted = styles.getPropertyValue("--muted").trim() || "#66727e";
  const line = styles.getPropertyValue("--line").trim() || "#d9d5ca";
  const accent = styles.getPropertyValue("--accent").trim() || "#1f6f5c";
  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = muted;
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;

  const points = state.solutions
    .map((s) => ({
      x: Number(s.complexity),
      train: Number(s.train_error),
      test: Number(s.test_error),
      rank: s.rank,
    }))
    .filter((p) => Number.isFinite(p.x) && p.x >= 0);
  const positive = (v) => Number.isFinite(v) && v > 0;
  const values = [];
  points.forEach((p) => {
    if (positive(p.train)) values.push(p.train);
    if (positive(p.test)) values.push(p.test);
  });
  if (state.baselines) {
    ["mean", "linear", "knn"].forEach((key) => {
      if (positive(state.baselines[key].test_error)) values.push(state.baselines[key].test_error);
    });
  }

  if (points.length === 0) {
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.fillText("points appear as solution events arrive", W / 2, H / 2);
    return;
  }

  let yMin = values.length ? Math.min(...values) : 1e-12;
  let yMax = values.length ? Math.max(...values) : 1;
  if (!Number.isFinite(yMin) || yMin <= 0) yMin = 1e-12;
  if (!Number.isFinite(yMax) || yMax <= yMin) yMax = yMin * 10;
  const yLo = Math.floor(Math.log10(yMin));
  const yHi = Math.ceil(Math.log10(yMax));
  const xMin = 0;
  const xMax = Math.max(1, ...points.map((p) => p.x));
  const sx = (x) => pad.left + (xMax === xMin ? 0.5 : (x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => {
    const v = Math.log10(Math.max(y, 1e-12));
    const lo = yLo === yHi ? yLo - 1 : yLo;
    const hi = yLo === yHi ? yHi + 1 : yHi;
    return pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;
  };

  // Grid and labels.
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const lo = yLo === yHi ? yLo - 1 : yLo;
  const hi = yLo === yHi ? yHi + 1 : yHi;
  const ticks = Math.min(6, hi - lo + 1);
  for (let i = 0; i < ticks; i += 1) {
    const power = lo + ((hi - lo) * i) / Math.max(1, ticks - 1);
    const y = pad.top + plotH - ((power - lo) / (hi - lo)) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = muted;
    ctx.fillText("1e" + Math.round(power), pad.left - 8, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = 5;
  for (let i = 0; i < xTicks; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / (xTicks - 1);
    const x = sx(value);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.fillStyle = muted;
    ctx.fillText(String(Math.round(value * 100) / 100), x, pad.top + plotH + 8);
  }
  ctx.fillStyle = ink;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("complexity", pad.left + plotW / 2, pad.top + plotH + 24);
  ctx.save();
  ctx.translate(16, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("error (log)", 0, 0);
  ctx.restore();

  // Baseline rules.
  if (state.baselines) {
    const labels = [
      ["mean", "constant mean"],
      ["linear", "linear"],
      ["knn", "k-NN"],
    ];
    ctx.setLineDash([5, 4]);
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    labels.forEach(([key, label]) => {
      const value = state.baselines[key].test_error;
      if (!positive(value)) return;
      const y = sy(value);
      ctx.strokeStyle = accent;
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = muted;
      ctx.fillText(label, pad.left + plotW - 2, y - 2);
    });
    ctx.setLineDash([]);
  }

  // Points: solid train, hollow test; a ring marks the selection.
  points.forEach((p, index) => {
    if (positive(p.train)) {
      ctx.beginPath();
      ctx.fillStyle = accent;
      ctx.arc(sx(p.x), sy(p.train), 4, 0, Math.PI * 2);
      ctx.fill();
    }
    if (positive(p.test)) {
      ctx.beginPath();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.5;
      ctx.arc(sx(p.x), sy(p.test), 4, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (index === state.selected) {
      ctx.beginPath();
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.5;
      ctx.arc(sx(p.x), sy(positive(p.train) ? p.train : p.test), 10, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
  });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.tab === name ? "true" : "false");
  });
  document.querySelectorAll(".tabpanel").forEach((panel) => {
    panel.hidden = panel.id !== "tab-" + name;
    panel.classList.toggle("active", panel.id === "tab-" + name);
  });
  if (name === "run") drawPareto();
}

async function loadExample() {
  try {
    const response = await fetch("/api/example");
    if (!response.ok) throw new Error("HTTP " + response.status);
    const text = await response.text();
    $("dataText").value = text.trim();
    state.fileName = "";
    applyTable(text);
  } catch (e) {
    setMessage("dataMessage", "cannot load example: " + e.message, "error");
  }
}

async function loadSchema() {
  const response = await fetch("/api/schema");
  if (!response.ok) throw new Error("HTTP " + response.status);
  state.schema = await response.json();
  renderForm();
  $("engineVersion").textContent = "rcsr " + state.schema.version;
  updateCli();
}

function wire() {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("parseBtn").addEventListener("click", () => {
    try {
      applyTable($("dataText").value);
    } catch (e) {
      setMessage("dataMessage", e.message, "error");
    }
  });
  let timer = null;
  $("dataText").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if ($("dataText").value.trim()) {
        try {
          applyTable($("dataText").value);
        } catch (e) {
          setMessage("dataMessage", e.message, "error");
        }
      }
    }, 450);
  });
  $("fileInput").addEventListener("change", async () => {
    const file = $("fileInput").files[0];
    if (!file) return;
    const text = await file.text();
    state.fileName = file.name;
    $("dataText").value = text;
    try {
      applyTable(text);
    } catch (e) {
      setMessage("dataMessage", e.message, "error");
    }
  });
  $("exampleBtn").addEventListener("click", loadExample);
  $("runBtn").addEventListener("click", startRun);
  $("copyCli").addEventListener("click", async () => {
    const text = copyText();
    if (!text) {
      setMessage("formMessage", "choose a target and at least one input first", "error");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setMessage("formMessage", "command copied", "ok");
    } catch (e) {
      setMessage("formMessage", "clipboard blocked; select the CLI bar text instead", "error");
    }
  });
  window.addEventListener("resize", drawPareto);
}

async function boot() {
  wire();
  try {
    await loadSchema();
  } catch (e) {
    setMessage("formMessage", "cannot read /api/schema: " + e.message, "error");
    setStatus("schema error", "error");
    return;
  }
  await loadExample();
  drawPareto();
  setStatus("ready", "ready");
}

boot();
