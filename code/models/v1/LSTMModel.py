import json
import polars as pl
from typing import Tuple, Any
from darts import TimeSeries
from darts.models import BlockRNNModel
from ..BaseModel import BaseModel
from matplotlib import pyplot as plt
from ..LossHistory import LossHistory
from darts.dataprocessing.transformers import Scaler
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from darts.metrics import mse, rmse, mae, r2_score
import datetime

class LSTMModel(BaseModel):
    def __init__(self, model="LSTM", input_chunk_length=2, output_chunk_length=1, hidden_dim=4, n_rnn_layers=1, batch_size=16, n_epochs=100, dropout=0.5):
        super().__init__("RNN - LSTM")
        self.loss_history = LossHistory()
        self.eary_stopping = EarlyStopping(
            mode="min",
            patience=30,
            monitor="val_loss"
        )
        self.base_images_path = "images/models/current"
        
        self.config_params = {
            "model_type": model,
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_dim": hidden_dim,
            "n_rnn_layers": n_rnn_layers,
            "batch_size": batch_size,
            "dropout": dropout,
            "n_epochs": n_epochs
        }
        
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
                "callbacks":[self.loss_history, self.eary_stopping]
            }
        )
        self.meta_cols = [
            "sensor_file", "timestamp", "time", "ToolIdx", "plate_id", "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", 
            "timestamp_right", "y"
        ]
        self.scaler_x = Scaler()
        self.scaler_y = Scaler()
        self.train_split, self.val_split = 0.60, 0.80
        self.res_mse = 0.0
        self.res_rmse = 0.0
        self.res_mae = 0.0
        self.res_r2 = 0.0
        self.fig_loss = None
        self.fig_pred = None
        self.Y_test_raw = None
        self.pred_df = None

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
        
        self.fig_loss = plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history.train_losses, label="Train Loss")
        plt.plot(self.loss_history.val_losses, label="Validation Loss")
        plt.legend()
        
        self.test(X, Y)
        self.save()
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
        self.pred_df = pl.from_pandas(pred_ts.to_dataframe().reset_index())
        self.Y_test_raw = Y[test_start:]
        
        self.res_mse = mse(self.Y_test_raw, pred_ts)
        self.res_rmse = rmse(self.Y_test_raw, pred_ts)
        self.res_mae = mae(self.Y_test_raw, pred_ts)
        self.res_r2 = r2_score(self.Y_test_raw, pred_ts)
        
        self.fig_pred, ax = plt.subplots(figsize=(12, 7))
        ax.plot(self.Y_test_raw.to_series().values, label="Vrai couple", color="black", linewidth=1.5)
        ax.plot(self.pred_df["y"].to_numpy(), label="Couple prédit (t+1)", color="orange", linestyle="--", linewidth=1.5)
        
        ax.set_title("Test Set", fontsize=14, fontweight='bold')
        ax.set_xlabel("Passes", fontsize=12)
        ax.set_ylabel("Couple (Nm)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        stats_text = (
            f"metrics :\n"
            f"-------------------\n"
            f"MSE  : {self.res_mse:.3f} Nm\n"
            f"RMSE : {self.res_rmse:.3f} Nm\n"
            f"MAE  : {self.res_mae:.3f} Nm\n"
            f"R²   : {self.res_r2:.3f}\n\n"
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
        return self.pred_df

    
    def save(self, path="./"):
        run_path = self._save()
        return super().save(run_path)
    def _save(self) -> None:
        run_name = f"run_{self.get_formated_datetime()}_{self.config_params['model_type']}"
        run_path = Path(self.base_images_path) / run_name
        run_path.mkdir(parents=True,exist_ok=True)

        if self.fig_loss is not None:
            self.fig_loss.savefig(run_path / "loss_curve.png", dpi=150)
            plt.close(self.fig_loss)
        
        if self.fig_pred is not None:
            self.fig_pred.savefig(run_path /"predictions_test.png", dpi=150)
            plt.close(self.fig_pred)
        plt.close()

        summary_report = {
            "experiment_timestamp": datetime.datetime.now().isoformat(),
            "model_hyperparameters": self.config_params,
            "dataset_splits": {
                "train_percentage": self.train_split * 100,
                "val_percentage": (self.val_split - self.train_split) * 100,
                "test_percentage": (1 - self.val_split) * 100
            },
            "evaluation_metrics": {
                "MSE": float(self.res_mse),
                "RMSE": float(self.res_rmse),
                "MAE": float(self.res_mae),
                "R2_Score": float(self.res_r2)
            }
        }
        
        with open(run_path /"metrics_config.json", "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        return run_path

    def preprocess_to_darts(self, df: pl.LazyFrame, target_column):
        df_cleansed = df.collect().to_dummies(columns=["DB_PASSES/SELECTION_ALLIAGE", "pass_type"])
        df_pd = df_cleansed.to_pandas()
        feature_cols = [c for c in df_cleansed.columns if c not in self.meta_cols]
        X = TimeSeries.from_dataframe(df_pd, value_cols=feature_cols)
        Y = TimeSeries.from_dataframe(df_pd, value_cols=target_column)
        return X, Y

    def load(self, path: Path) -> None:
        model_file = path / f"{self.model_name}.pt"
        if model_file.exists():
            self.model = BlockRNNModel.load(model_file)
        else:
            raise FileNotFoundError(f"No checkpoint found at: {model_file}")