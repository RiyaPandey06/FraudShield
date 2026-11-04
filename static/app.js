/* FraudShield analyst console: React 18 (UMD, no build step). */
const { useState, useEffect, useRef, useCallback } = React;
const h = React.createElement;
const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* keep status text */ }
    throw new Error(detail);
  }
  return response.json();
}

// Deterministic arm assignment per case for the review-time study (explanation text shown or hidden).
function armFor(id) {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  return hash % 2 === 0 ? "explained" : "raw";
}

function Metric(props) {
  return h("div", { className: "metric" }, h("span", null, props.label), h("strong", null, props.value));
}

function DriftBanner(props) {
  const drift = props.summary.drift;
  const last = props.summary.retrains && props.summary.retrains[0];
  if (!drift) return h("div", { className: "banner ok" }, "Drift monitor is warming up (needs a full window of live transactions).");
  const top = drift.results.slice().sort((a, b) => b.psi - a.psi).slice(0, 3)
    .map((r) => r.feature_name + " PSI " + r.psi.toFixed(2)).join(" · ");
  const drifting = drift.drifted.length > 0;
  return h("div", { className: "banner " + (drifting ? "warn" : "ok") },
    h("span", null, (drifting ? "Drift watch: " : "Stable: ") + top +
      (last ? "  |  last retrain: " + last.trigger + " -> " + last.outcome : "")),
    h("button", { className: "secondary", onClick: props.onRetrain, disabled: props.busy }, props.busy ? "Retraining..." : "Retrain now"));
}

function CaseDetail(props) {
  const { alert, study, onLabel } = props;
  const openedAt = useRef(Date.now());
  useEffect(() => { openedAt.current = Date.now(); }, [alert.transaction_id]);
  const arm = study ? armFor(alert.transaction_id) : null;
  const showText = !study || arm === "explained";
  const linked = alert.graph.linked_account_ids || [];
  const submit = (label) => onLabel(alert.transaction_id, label, Date.now() - openedAt.current, arm);
  return h("div", { className: "detail" },
    h("h3", null, alert.transaction_id + " ", h("span", { className: "score" }, (alert.risk_score * 100).toFixed(0) + "%")),
    showText
      ? h("div", { className: "expl" }, alert.explanation,
          h("div", { className: "src" }, "text source: " + (alert.explanation_source === "llm" ? "local LLM (grounded)" : "template")))
      : h("div", { className: "expl muted" }, "Explanation text hidden (study mode, arm: raw reasons only)."),
    h("h4", null, "Top model reasons (always shown)"),
    alert.reasons.map((r) => h("div", { className: "reason", key: r.feature },
      h("span", null, r.label, h("br"), h("small", { className: "muted" }, r.feature + " = " + r.value)),
      h("b", { className: r.contribution < 0 ? "neg" : "" }, (r.contribution > 0 ? "+" : "") + r.contribution))),
    h("div", { className: "data" },
      h("div", null, "Amount", h("br"), h("code", null, money.format(alert.amount))),
      h("div", null, "Country", h("br"), h("code", null, alert.country)),
      h("div", null, "Account", h("br"), h("code", null, alert.account_id)),
      h("div", null, "Device", h("br"), h("code", null, alert.device_id)),
      h("div", null, "Accounts sharing devices", h("br"), h("code", null, alert.graph.linked_accounts)),
      h("div", null, "Identity cluster size", h("br"), h("code", null, alert.graph.community_size)),
      h("div", null, "Ring flag", h("br"), h("code", null, alert.graph.ring_flag ? "yes" : "no")),
      h("div", null, "Mule device flag", h("br"), h("code", null, alert.graph.mule_flag ? "yes" : "no"))),
    h("div", { className: "links" }, "Linked accounts: " + (linked.length ? linked.join(", ") : "none"),
      h("br"), "Merchant: " + alert.merchant_id),
    h("div", { className: "feedback" },
      h("button", { onClick: () => submit("fraud") }, "Confirm fraud"),
      h("button", { className: "secondary", onClick: () => submit("legitimate") }, "Mark legitimate")));
}

function App() {
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const [study, setStudy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const load = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([api("/api/summary"), api("/api/alerts?limit=100")]);
      setSummary(s); setAlerts(a); setError("");
    } catch (e) { setError("Unable to reach API: " + e.message); }
  }, []);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);

  const open = async (id) => {
    try { setSelected(await api("/api/alerts/" + encodeURIComponent(id))); } catch (e) { setError(e.message); }
  };
  const label = async (id, value, reviewMs, arm) => {
    try {
      await api("/api/alerts/" + encodeURIComponent(id) + "/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: value, review_ms: reviewMs, arm: arm }),
      });
      setSelected(null); setNotice("Saved " + value + " label for " + id + " (" + (reviewMs / 1000).toFixed(1) + "s review)."); load();
    } catch (e) { setError(e.message); }
  };
  const retrain = async () => {
    setBusy(true);
    try { const r = await api("/api/retrain", { method: "POST" }); setNotice("Retrain outcome: " + r.outcome); load(); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };

  const m = summary ? summary.model_metrics : null;
  return h("div", null,
    h("header", null,
      h("div", null, h("h1", null, "FraudShield"), h("p", { className: "sub" }, "Real-time fraud analyst console · graph-aware risk scoring")),
      h("div", { className: "pill" }, summary ? "Model " + summary.model_version + " · " + summary.backend + " · " + summary.explainer : "Loading model...")),
    h("main", null,
      summary && h("section", { className: "metrics" },
        h(Metric, { label: "Transactions processed", value: summary.transactions_processed.toLocaleString() }),
        h(Metric, { label: "Open alerts", value: summary.open_alerts.toLocaleString() }),
        h(Metric, { label: "PR-AUC (test)", value: (m.pr_auc * 100).toFixed(1) + "%" }),
        h(Metric, { label: "Cost threshold", value: summary.threshold.toFixed(3) }),
        h(Metric, { label: "Labels matured / feedback", value: summary.labels_matured + " / " + summary.feedback_count })),
      summary && h(DriftBanner, { summary: summary, onRetrain: retrain, busy: busy }),
      error && h("div", { className: "banner warn" }, error),
      notice && h("div", { className: "banner ok" }, notice),
      h("section", { className: "layout" },
        h("div", { className: "card" },
          h("h2", null, "Prioritised alert queue"),
          h("div", { className: "toolbar" }, h("span", null, alerts.length + " open alert" + (alerts.length === 1 ? "" : "s")),
            h("label", { className: "chk" }, h("input", { type: "checkbox", checked: study, onChange: (e) => setStudy(e.target.checked) }),
              "Review-time study mode (randomly hides explanation text)"),
            h("button", { className: "secondary", onClick: load }, "Refresh")),
          h("div", { className: "table" }, h("table", null,
            h("thead", null, h("tr", null, ["Risk", "Transaction", "Amount", "Account", "Time"].map((c) => h("th", { key: c }, c)))),
            h("tbody", null, alerts.length ? alerts.map((a) => h("tr", {
              key: a.transaction_id, className: "row" + (selected && selected.transaction_id === a.transaction_id ? " sel" : ""),
              onClick: () => open(a.transaction_id) },
              h("td", { className: "score" }, (a.risk_score * 100).toFixed(0) + "%"), h("td", null, a.transaction_id),
              h("td", null, money.format(a.amount)), h("td", null, a.account_id), h("td", null, new Date(a.timestamp).toLocaleString())))
              : h("tr", null, h("td", { colSpan: 5, className: "empty" }, "No open alerts yet. Start the replay to populate this queue.")))))),
        h("aside", { className: "card" }, h("h2", null, "Case detail"),
          selected ? h(CaseDetail, { alert: selected, study: study, onLabel: label })
                   : h("div", { className: "empty" }, "Select an alert to see model reasons, graph links and the explanation.")))));
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
