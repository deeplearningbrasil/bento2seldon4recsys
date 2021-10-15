from typing import Type, cast

import random

import bento2seldon
from bento2seldon.bento import BaseRouter
from bento2seldon.seldon import SeldonMessageRequest
from bentoml import env, ver

from bento2seldon4recsys.model import RankingRequest
from bento2seldon4recsys.router.ab_test.model import ABTestSettings


@ver(major=1, minor=0)
@env(
    conda_channels=["conda-forge"],
    conda_dependencies=["redis-py>=3.5"],
    pip_packages=[f"bento2seldon=={bento2seldon.__version__}"],  # type: ignore[attr-defined]
)
class ABTestRouter(BaseRouter[RankingRequest]):
    @property
    def settings(self) -> ABTestSettings:
        if not hasattr(self, "_settings"):
            self._settings = ABTestSettings()
        return cast(ABTestSettings, self._settings)

    @property
    def request_type(self) -> Type[RankingRequest]:
        return RankingRequest

    def _route(self, seldon_message: SeldonMessageRequest[RankingRequest]) -> int:
        if random.random() <= self.settings.b_ratio:
            return 1
        else:
            return 0
