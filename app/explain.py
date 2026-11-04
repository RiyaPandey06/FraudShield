"""Analyst-facing explanations.

Every alert always carries (a) the raw top-3 contributions and (b) a deterministic template sentence.
An optional local LLM (Ollama) rewrites that into smoother prose *asynchronously*; its output is
accepted only if it passes a grounding check (no numbers that are not in the supplied facts), otherwise
the template is kept. The LLM never sees anything except the supplied facts.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def template_explanation(score: float, reasons: list[dict], graph: dict, decision: str) -> str:
    risky = [r["label"] for r in reasons if r["direction"] == "increases_risk"]
    reason_text = ", ".join(risky[:2]) or "the model's overall transaction pattern"
    if decision == "allow":
        return f"Allowed at the configured threshold. The main risk signals were {reason_text}."
    graph_text = ""
    if graph.get("linked_accounts"):
        graph_text = f" The account is linked to {graph['linked_accounts']} other account(s) through shared devices."
    elif graph.get("community_size", 1) > 4:
        graph_text = f" The account is part of a {graph['community_size']}-account identity cluster."
    return (f"Flagged with {score:.0%} risk because of {reason_text}.{graph_text} "
            "Review the raw feature reasons before taking action.")


def facts_for(alert: dict) -> list[str]:
    """The complete, closed set of facts the LLM may use."""
    facts = [f"Risk score: {alert['risk_score']:.0%}", f"Amount: {alert['amount']:.2f}", f"Country: {alert['country']}"]
    for reason in alert["reasons"]:
        facts.append(f"Model signal ({reason['direction'].replace('_', ' ')}): {reason['label']} "
                     f"({reason['feature']} = {reason['value']})")
    graph = alert["graph"]
    facts.append(f"Linked accounts via shared devices: {graph.get('linked_accounts', 0)}")
    facts.append(f"Identity cluster size (accounts): {graph.get('community_size', 1)}")
    return facts


def is_grounded(text: str, facts: list[str]) -> bool:
    allowed = set()
    for fact in facts:
        for number in NUMBER.findall(fact):
            allowed.add(number)
            allowed.add(number.rstrip("0").rstrip(".") if "." in number else number)
    return all(n in allowed or n.rstrip("0").rstrip(".") in allowed for n in NUMBER.findall(text))


class OllamaExplainer:
    def __init__(self, url: str, model: str, timeout: float = 20.0) -> None:
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def prompt(self, facts: list[str]) -> str:
        return ("You help a fraud analyst. Using ONLY the facts below, write 2 short sentences explaining why "
                "this payment was flagged. Do not invent facts, do not state that fraud occurred, do not use any "
                "number that is not listed.\n\nFacts:\n- " + "\n- ".join(facts))

    def explain(self, alert: dict) -> str | None:
        facts = facts_for(alert)
        body = json.dumps({"model": self.model, "prompt": self.prompt(facts), "stream": False,
                           "options": {"temperature": 0.1, "num_predict": 120}}).encode()
        request = urllib.request.Request(f"{self.url}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = json.loads(response.read())["response"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError):
            return None
        return text if text and is_grounded(text, facts) else None
