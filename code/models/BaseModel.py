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
    def __init__(self,data:pl.LazyFrame,target_column:str, model_name: str):
        self.model_name = model_name
        self.model = None
        self.data:pl.LazyFrame = data
        self.target_column:str = target_column

    @abstractmethod
    def preprocess_to_darts(self, df: pl.DataFrame, target_column: str):
        pass

    @abstractmethod
    def train(self, X: TimeSeries, Y: Any = None):
        pass

    @abstractmethod
    def infer(self, X: TimeSeries, n_steps: int = 1) -> pl.DataFrame:
        pass
    
    def start_training(self,saving=True) -> str:
        X, Y = self.preprocess_to_darts(self.data, target_column=self.target_column)
        model_path = self.train(X, Y)
        if saving:
            self.save()
        return model_path
    
    @abstractmethod
    def save(self) -> Path:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass