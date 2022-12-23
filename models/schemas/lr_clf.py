from pydantic import BaseModel, conlist
from typing import List, Any


class FSI(BaseModel):
    data: List[conlist(float, min_items=8, max_items=8)]

class FSIPredictionResponse(BaseModel):
    prediction: List[int]
    probability: List[Any]
    log_probability: List[Any]

class FSIModelResponse(BaseModel):
    prediction: List[Any]
    f1_score: List[float]
    roc_auc_score: List[float]
    y_proba: List[Any]