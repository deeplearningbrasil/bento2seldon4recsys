from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import abc
import datetime
import logging

from bento2seldon.adapter import SeldonJsonInput
from bento2seldon.bento import (
    BaseBatchPredictor,
    BasePredictor,
    BaseSinglePredictor,
    ExceptionHandler,
    I,
)
from bento2seldon.cache import Cache
from bento2seldon.logging import LoggingContext
from bento2seldon.seldon import PRED_UNIT_ID, PRED_UNIT_KEY, Meta, SeldonMessage
from bentoml import api
from bentoml.types import InferenceTask
from pydantic import parse_obj_as

from bento2seldon4recsys.model import (
    RankingRequest,
    RankingResponse,
    RecommenderSettings,
)
from bento2seldon4recsys.monitoring import RecommenderMonitor
from bento2seldon4recsys.ranking_metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
)

REQUEST_KEY = "request"

logger = logging.getLogger(__name__)

RT = TypeVar("RT", bound=RankingRequest)
RE = TypeVar("RE", bound=RankingResponse)


class RecommenderCache(Cache[RT, RE], Generic[RT, RE]):
    def should_cache(self, request: RT, response: RE, meta: Meta) -> bool:
        return super().should_cache(request, response, meta) and bool(response.item_ids)


class _BaseRecommenderMixin(Generic[RT, RE]):
    @property
    def request_type(self) -> Type[RT]:
        return RankingRequest  # type: ignore[return-value]

    @property
    def response_type(self) -> Type[RE]:
        return RankingResponse  # type: ignore[return-value]

    @property
    def settings(self) -> RecommenderSettings:
        if not hasattr(self, "_settings"):
            self._settings = RecommenderSettings()
        return self._settings

    def _parse_input(
        self,
        raw_request: Union[Dict[str, Any], List[Dict[str, Any]]],
        task: InferenceTask,
        request_type: Type[I],
        logging_context: LoggingContext,
    ) -> Optional[I]:
        input_ = cast(BasePredictor[RT, RE], super())._parse_input(
            raw_request, task, request_type, logging_context
        )
        if (
            self.settings.is_cold_start_recommender_child
            and isinstance(input_, SeldonMessage)
            and input_.jsonData is not None
        ):
            input_.meta.tags[REQUEST_KEY] = input_.jsonData.dict(exclude_none=True)
        return input_


class _FeedbackMixin(Generic[RT, RE]):
    @property
    def cache(  # type: ignore[misc]
        self: BasePredictor[RT, RE]
    ) -> RecommenderCache[RT, RE]:
        if not hasattr(self, "_cache"):
            self._cache = RecommenderCache[RT, RE](
                self,
                self.request_type,
                self.response_type,
                self.settings.redis_url,
                datetime.timedelta(seconds=self.settings.cache_duration),
            )
        return cast(RecommenderCache[RT, RE], self._cache)

    @property
    def monitor(self) -> RecommenderMonitor:
        if not hasattr(self, "_monitor"):
            self._monitor = RecommenderMonitor(self)
        return self._monitor

    def _should_threat_feedback(self, response: SeldonMessage[RE]) -> bool:
        return (
            response.meta.tags.get(PRED_UNIT_KEY) == PRED_UNIT_ID
            and response.jsonData is not None
            and bool(response.jsonData.item_ids)
        )

    def _send_feedback(
        self,
        request: Optional[SeldonMessage[RT]],
        response: Optional[SeldonMessage[RE]],
        truth: Optional[SeldonMessage[RE]],
        reward: Optional[float],
        routing: Optional[int],
    ) -> None:
        if response is not None and self._should_threat_feedback(response):
            logger.debug("Threating the feedback...")
            cast(BasePredictor[RT, RE], super())._send_feedback(
                request, response, truth, reward, routing
            )

            if (
                truth is not None
                and truth.jsonData is not None
                and request is not None
                and request.jsonData is not None
                and response.jsonData is not None
            ):
                true_item_ids = set(truth.jsonData.item_ids)
                logger.debug("true_item_ids: %s", true_item_ids)
                relevance_scores = [
                    int(item_id in true_item_ids)
                    for item_id in response.jsonData.item_ids
                ]
                logger.debug("relevance_scores: %s", relevance_scores)

                ks = [request.jsonData.top_k]
                if 10 < request.jsonData.top_k:
                    ks.append(10)
                if 50 < request.jsonData.top_k:
                    ks.append(50)

                for k in set(ks):
                    logger.debug("Calculating metrics for k=%d", k)
                    self.monitor.observe_ndcg(ndcg_at_k(relevance_scores, k), k)
                    self.monitor.observe_precision(
                        precision_at_k(relevance_scores, k), k
                    )
                self.monitor.observe_average_precision(
                    average_precision(relevance_scores)
                )


class BaseSingleRecommender(  # type: ignore[misc]
    _FeedbackMixin[RT, RE],
    _BaseRecommenderMixin[RT, RE],
    BaseSinglePredictor[RT, RE],
    Generic[RT, RE],
    metaclass=abc.ABCMeta,
):
    pass


class BaseBatchRecommender(  # type: ignore[misc]
    _FeedbackMixin[RT, RE],
    _BaseRecommenderMixin[RT, RE],
    BaseBatchPredictor[RT, RE],
    Generic[RT, RE],
    metaclass=abc.ABCMeta,
):
    pass


class BaseColdStartRecommender(  # type: ignore[misc]
    _FeedbackMixin[RT, RE],
    BasePredictor[RT, RE],
    Generic[RT, RE],
    metaclass=abc.ABCMeta,
):
    @property
    def request_type(self) -> Type[RT]:
        return RankingRequest  # type: ignore[return-value]

    @property
    def response_type(self) -> Type[RE]:
        return RankingResponse  # type: ignore[return-value]

    def _merge_meta(self, metas: List[Meta]) -> Meta:
        tags = {}
        for meta in metas:
            if meta:
                tags.update(meta.tags)
        return Meta(puid=metas[0].puid, tags=tags)

    @abc.abstractmethod
    def _predict_for_cold_start(self, request: RT) -> RE:
        pass

    @api(input=SeldonJsonInput(), batch=False)
    def aggregate(
        self, raw_seldon_message_list: List[Dict[str, Any]], task: InferenceTask = None
    ) -> Optional[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="aggregate")
        logger.debug("/aggregate: %s", raw_seldon_message_list)

        if task is None:
            task = InferenceTask()

        with self.monitor.count_exceptions(endpoint="combine"), ExceptionHandler(
            [task], logging_context
        ):
            seldon_message_list: Optional[List[SeldonMessage[RE]]] = self._parse_input(
                raw_seldon_message_list,
                task,
                List[SeldonMessage[self.response_type]],  # type: ignore[name-defined]
                logging_context,
            )

            if seldon_message_list is not None:
                assert (
                    len(seldon_message_list) == 1
                ), "The cold start recommender is not a regular combiner. It should be on top of only one model."

                seldon_message_response = seldon_message_list[0]
                meta = seldon_message_response.meta

                if seldon_message_response.jsonData is None:
                    logger.warning(
                        "Empty response"
                    )  # workaround for https://github.com/SeldonIO/seldon-core/issues/3139
                    return self._format_response(None, meta)

                request = parse_obj_as(self.request_type, meta.tags.pop(REQUEST_KEY))

                response = self.cache.get_response(
                    seldon_message_response.meta.puid, request
                )

                if response is None:
                    if seldon_message_response.jsonData.item_ids:
                        logger.debug("Returning the original response")
                        response = seldon_message_response.jsonData
                    else:
                        logger.debug("Returning a cold start response")
                        response = self._predict_for_cold_start(request)
                        meta = self._merge_meta([meta, Meta()])
                        self.cache.set_response(request, response, meta)

                return self._format_response(response, meta)
            else:
                return None
