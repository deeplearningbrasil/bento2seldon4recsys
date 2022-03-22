from typing import Dict, List, Optional

from bento2seldon.monitoring import FEEDBACK_ENDPOINT, Monitor
from bento2seldon.seldon import DEPLOYMENT_ID
from bentoml import BentoService
from prometheus_client.metrics import Histogram


class RecommenderMonitor(Monitor):
    def __init__(
        self,
        bento_service: BentoService,
        extra_labels_per_metric: Optional[Dict[str, List[str]]] = {},
    ) -> None:
        super().__init__(bento_service, extra_labels_per_metric)

        self._ndcg = self._create_metric(
            Histogram,
            "ndcg",
            "nDCG@k",
            ["k"],
        )

        self._precision = self._create_metric(
            Histogram,
            "precision",
            "precision@k",
            ["k"],
        )

        self._average_precision = self._create_metric(
            Histogram,
            "average_precision",
            "_average_precision",
        )

    def observe_ndcg(
        self,
        value: float,
        k: int,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra_labels_values: list = [],
    ) -> None:
        self._ndcg.labels(
            DEPLOYMENT_ID, self.version, endpoint, *extra_labels_values, k
        ).observe(value)

    def observe_precision(
        self,
        value: float,
        k: int,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra_labels_values: list = [],
    ) -> None:
        self._precision.labels(
            DEPLOYMENT_ID, self.version, endpoint, *extra_labels_values, k
        ).observe(value)

    def observe_average_precision(
        self,
        value: float,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra_labels_values: list = [],
    ) -> None:
        self._average_precision.labels(
            DEPLOYMENT_ID, self.version, endpoint, *extra_labels_values
        ).observe(value)
