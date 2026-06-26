import json
import polars as pl
from typing import Tuple, Any
from darts import TimeSeries
from darts.models import TFTModel as DartsTFTModel
from .BaseModel import BaseModel 
from matplotlib import pyplot as plt
from .LossHistory import LossHistory
from darts.dataprocessing.transformers import Scaler
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from darts.metrics import mse, rmse, mae, r2_score
import datetime
import optuna
from optuna.trial import TrialState

class TFTModel(BaseModel):
    def __init__(self, data: pl.LazyFrame, train_tools, val_tools, test_tools,
                 input_chunk_length=40, output_chunk_length=1, hidden_size=64,
                 lstm_layers=1, num_attention_heads=4, batch_size=32, n_epochs=100,
                 dropout=0.1, filepath="", meta_cols=[], target_column="y",
                 learning_rate=1e-3, base_image_path=""):
        self.loss_history = LossHistory()
        self.early_stopping = EarlyStopping(
            mode="min",
            patience=10,
            monitor="val_loss"
        )
        self.base_images_path = base_image_path
        self.filepath = filepath
        self.features_cols = []
        self.meta_cols = meta_cols
        self.target_col = target_column
        self.data:pl.LazyFrame = data
        self.train_tools = train_tools
        self.test_tools = test_tools
        self.val_tools = val_tools
        
        self.config_params = {
            "model_type": "TFT",
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "num_attention_heads": num_attention_heads,
            "batch_size": batch_size,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate
        }

        
        
        self.model = DartsTFTModel(
            input_chunk_length=self.config_params["input_chunk_length"],
            output_chunk_length=self.config_params["output_chunk_length"],
            hidden_size=self.config_params["hidden_size"],
            lstm_layers=self.config_params["lstm_layers"],
            num_attention_heads=self.config_params["num_attention_heads"],
            batch_size=self.config_params["batch_size"],
            dropout=self.config_params["dropout"],
            n_epochs=self.config_params["n_epochs"],
            add_relative_index=True,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "callbacks": [self.loss_history, self.early_stopping],
                "enable_progress_bar": True,
                "enable_model_summary": False
            },
            optimizer_kwargs={"lr": self.config_params["learning_rate"]}
        )
        self.meta_cols = self.meta_cols
        self.scaler_x = Scaler()
        self.scaler_y = Scaler()
        self.res_mse = 0.0
        self.res_rmse = 0.0
        self.res_mae = 0.0
        self.res_r2 = 0.0
        self.fig_loss = None
        self.fig_pred = None
        self.Y_test_raw = None
        self.pred_df = None
        self.filepath = None

    @staticmethod
    def get_formated_datetime():
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    