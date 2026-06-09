import os
import polars as pl
from typing import Tuple, Any
from darts import TimeSeries
from darts.models import BlockRNNModel
from .BaseModel import BaseModel
from matplotlib import pyplot as plt
from .LossHistory import LossHistory
from darts.dataprocessing.transformers import Scaler
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from darts.metrics import mse, rmse, mae, r2_score
import datetime

class LSTMModel(BaseModel):
    def __init__(self, model="LSTM", input_chunk_length=5, output_chunk_length=1, hidden_dim=8, n_rnn_layers=1, batch_size=16, n_epochs=100, dropout=0.1):
        super().__init__("RNN - LSTM")
        self.loss_history = LossHistory()
        self.eary_stopping = EarlyStopping(
            mode="min",
            patience=10,
            monitor="val_loss"
        )
        self.images_path = "images/models/current"
        
        self.model = BlockRNNModel(
            model=model,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            batch_size=batch_size,
            dropout=dropout,
            n_epochs=n_epochs,
            pl_trainer_kwargs={
                "callbacks":[self.loss_history]
            }
        )
        self.meta_cols = [
            "sensor_file", "timestamp", "time", "ToolIdx", "PassNumber", "plate_id", "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", 
            "timestamp_right", "y"
        ]
        self.scaler_x = Scaler()
        self.scaler_y = Scaler()
        self.train_split, self.val_split = 0.70, 0.85
        
        os.makedirs(self.images_path, exist_ok=True)
        
    def get_formated_datetime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def train(self, X: TimeSeries, Y: TimeSeries = None):
        train_end = int(len(Y) * self.train_split)
        val_end = int(len(Y) * self.val_split)
        
        Y_train_raw = Y[:train_end]
        Y_val_raw = Y[train_end:val_end]
        
        X_train_raw = X[:train_end]
        X_val_raw = X[train_end:val_end]
        
        X_train = self.scaler_x.fit_transform(X_train_raw)
        Y_train = self.scaler_y.fit_transform(Y_train_raw)
        
        X_val = self.scaler_x.transform(X_val_raw)
        Y_val = self.scaler_y.transform(Y_val_raw)
        
        self.model.fit(
            series=Y_train, 
            future_covariates=X_train,
            val_series=Y_val,
            val_future_covariates=X_val
        )
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history.train_losses, label="Train Loss")
        plt.plot(self.loss_history.val_losses, label="Validation Loss")
        plt.legend()
        path_name=f"{self.images_path}/LSTM_loss_curve_{self.get_formated_datetime()}.png"
        plt.savefig(path_name, dpi=150)
        print(f"saved figure to {path_name}")
        plt.close()
        
        self.test(X, Y)
        return self.model_name

    def infer(self, X: TimeSeries, n_steps = 1):
        X_scaled = self.scaler_x.transform(X)
        pred_ts_scaled = self.model.predict(n=n_steps, future_covariates=X_scaled)
        pred_ts = self.scaler_y.inverse_transform(pred_ts_scaled)
        pred_df = pl.from_pandas(pred_ts.to_dataframe().reset_index())
        return pred_df

    def test(self, X: TimeSeries, Y: TimeSeries):
        test_start = int(len(Y) * self.val_split)
        
        X_scaled = self.scaler_x.transform(X)
        Y_scaled = self.scaler_y.transform(Y)
        
        pred_ts_scaled = self.model.historical_forecasts(
            series=Y_scaled,
            future_covariates=X_scaled,
            start=test_start,
            forecast_horizon=1,
            retrain=False,
            last_points_only=True
        )
        
        pred_ts = self.scaler_y.inverse_transform(pred_ts_scaled)
        pred_df = pl.from_pandas(pred_ts.to_dataframe().reset_index())
        Y_test_raw = Y[test_start:]
        
        res_mse = mse(Y_test_raw, pred_ts)
        res_rmse = rmse(Y_test_raw, pred_ts)
        res_mae = mae(Y_test_raw, pred_ts)
        res_r2 = r2_score(Y_test_raw, pred_ts)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(Y_test_raw.to_series().values, label="Vrai couple", color="black", linewidth=1.5)
        ax.plot(pred_df["y"].to_numpy(), label="Couple prédit (t+1)", color="orange", linestyle="--", linewidth=1.5)
        
        ax.set_title("Test Set", fontsize=14, fontweight='bold')
        ax.set_xlabel("Passes", fontsize=12)
        ax.set_ylabel("Couple (Nm)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        stats_text = (
            f"metrics :\n"
            f"-------------------\n"
            f"MSE  : {res_mse:.3f}\n"
            f"RMSE : {res_rmse:.3f} Nm\n"
            f"MAE  : {res_mae:.3f} Nm\n"
            f"R²   : {res_r2:.3f}\n\n"
            f"model confguration :\n"
            f"----------------------\n"
            f"Hidden Dim : {self.model.model_params['hidden_dim']}\n"
            f"Layers     : {self.model.model_params['n_rnn_layers']}\n"
            f"Lookback   : {self.model.model_params['input_chunk_length']} passes\n"
            f"Dropout    : {self.model.model_params['dropout']}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.97, 0.96, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')
        
        plt.tight_layout()
        path_name = f"{self.images_path}/LSTM_predictions_test_{self.get_formated_datetime()}.png"
        plt.savefig(path_name, dpi=150)
        print(f"saved figure to {path_name}")
        plt.close(fig)
        return pred_df

    def preprocess_to_darts(self, df: pl.LazyFrame, target_column):
        mapping_passes = {
            "Blanking": 0,
            "Roughing": 1,
            "Pre-Finishing": 2,
            "Finishing": 3
        }
        
        df_cleansed = df.with_columns(
            pl.col("pass_type").replace(mapping_passes).cast(pl.Int32),
        ).collect()
        df_cleansed = df_cleansed.to_dummies(columns=["DB_PASSES/SELECTION_ALLIAGE", "pass_type"])
        
        df_pd = df_cleansed.to_pandas()
        feature_cols = [c for c in df_cleansed.columns if c not in self.meta_cols]
        X = TimeSeries.from_dataframe(df_pd, value_cols=feature_cols)
        Y = TimeSeries.from_dataframe(df_pd, value_cols=target_column)
        return X, Y

    def load(self, path: Path) -> None:
        model_file = path / f"{self.model_name}.pt"
        if os.path.exists(model_file):
            self.model = BlockRNNModel.load(model_file)
        else:
            raise FileNotFoundError(f"No checkpoint found at: {model_file}")