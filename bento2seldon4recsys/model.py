from typing import List

from bento2seldon.model import Settings
from pydantic import BaseModel


class RankingRequest(BaseModel):
    user_id: str
    top_k: int = 10


class RankingResponse(BaseModel):
    item_ids: List[str]


class RecommenderSettings(Settings):
    is_cold_start_recommender_child: bool = False
