import os
import polars as pl
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Type
from darts import TimeSeries
import datetime
import os
import polars as pl
from abc import ABC, abstractmethod
from typing import Any, Tuple
from darts import TimeSeries
import datetime
from pathlib import Path

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def preprocess_to_darts(self, df: pl.DataFrame, target_column: str):
        pass

    @abstractmethod
    def train(self, X: TimeSeries, Y: Any = None):
        pass

    @abstractmethod
    def infer(self, X: TimeSeries, n_steps: int = 1) -> pl.DataFrame:
        pass
    
    def start_training(self, data: pl.DataFrame, target_column: str = "y",saving=False) -> str:
        X, Y = self.preprocess_to_darts(data, target_column=target_column)
        model_path = self.train(X, Y)
        if saving:
            self.save()
        return model_path

    def save(self, path: Path=Path()) -> None:
        if self.model is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model.save(str(path / f"{self.model_name}_{timestamp}"))
        else:
            raise ValueError("Error while saving model")

    @abstractmethod
    def load(self, path: str) -> None:
        pass