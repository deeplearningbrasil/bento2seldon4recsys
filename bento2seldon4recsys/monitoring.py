from typing import Any, Dict

from bento2seldon.monitoring import FEEDBACK_ENDPOINT, Monitor
from bento2seldon.seldon import DEPLOYMENT_ID
from bentoml import BentoService
from prometheus_client.metrics import Histogram


class RecommenderMonitor(Monitor):
    def __init__(self, bento_service: BentoService) -> None:
        super().__init__(bento_service)

    def observe_ndcg(
        self,
        value: float,
        k: int,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra: Dict[str, Any] = {},
    ) -> None:
        if not hasattr(self, "_ndcg"):
            self._ndcg = self._create_metric(
                Histogram,
                "ndcg",
                "nDCG@k",
                ["k", *extra.keys()],
            )

        self._ndcg.labels(
            DEPLOYMENT_ID, self.version, endpoint, k, *extra.values()
        ).observe(value)

    def observe_precision(
        self,
        value: float,
        k: int,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra: Dict[str, Any] = {},
    ) -> None:
        if not hasattr(self, "_precision"):
            self._precision = self._create_metric(
                Histogram, "precision", "precision@k", ["k", *extra.keys()]
            )

        self._precision.labels(
            DEPLOYMENT_ID, self.version, endpoint, k, *extra.values()
        ).observe(value)

    def observe_average_precision(
        self,
        value: float,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra: Dict[str, Any] = {},
    ) -> None:
        if not hasattr(self, "_average_precision"):
            self._average_precision = self._create_metric(
                Histogram, "average_precision", "_average_precision", extra.keys()
            )

        self._average_precision.labels(
            DEPLOYMENT_ID, self.version, endpoint, *extra.values()
        ).observe(value)
