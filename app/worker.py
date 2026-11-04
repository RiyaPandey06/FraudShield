"""Stream scorer: consumes transactions, scores them, upgrades explanations with the optional LLM,
exposes Prometheus metrics on :9100 and triggers drift-driven retraining."""
from __future__ import annotations

import json
import logging
import threading
import time

from kafka import KafkaConsumer

from app.config import get_settings
from app.explain import OllamaExplainer
from app.metrics import start_http_server
from app.services import ServiceBundle

log = logging.getLogger("fraudshield.worker")


def explanation_loop(bundle: ServiceBundle, explainer: OllamaExplainer) -> None:
    """Asynchronous 'explanation service': never blocks scoring; alerts keep their template text on failure."""
    while True:
        pending = bundle.database.alerts_needing_llm(limit=3)
        if not pending:
            time.sleep(2)
            continue
        for alert in pending:
            text = explainer.explain(alert)
            bundle.database.update_explanation(alert["transaction_id"], text, "llm")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    start_http_server(9100)
    bundle = ServiceBundle.create(settings)
    if settings.ollama_url:
        explainer = OllamaExplainer(settings.ollama_url, settings.ollama_model)
        threading.Thread(target=explanation_loop, args=(bundle, explainer), daemon=True).start()
        log.info("LLM explanations enabled via %s", settings.ollama_url)
    consumer = KafkaConsumer(
        "transactions", bootstrap_servers=settings.kafka_servers.split(","), group_id="fraudshield-scorer-v2",
        auto_offset_reset="earliest", enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")))
    log.info("FraudShield worker consuming; threshold=%.4f", bundle.threshold)
    for message in consumer:
        result = bundle.process(message.value)
        log.info("%s -> %s (%.3f, %.1f ms)", result["transaction_id"], result["decision"], result["risk_score"],
                 result["latency_ms"])


if __name__ == "__main__":
    main()
