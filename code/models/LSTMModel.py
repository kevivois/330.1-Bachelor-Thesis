import json
import polars as pl
from darts import TimeSeries
from darts.models import BlockRNNModel
from BaseModel import BaseModel
from matplotlib import pyplot as plt
from LossHistory import LossHistory
from darts.dataprocessing.transformers import Scaler
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from darts.metrics import mse, rmse, mae, r2_score
import datetime
import optuna
from optuna.trial import TrialState


'''
Class containing the function and the variables to train , test and infer an LSTM Model using 'darts' library and optuna to optimize hyperparameters
'''
class LSTMModel(BaseModel):
    def __init__(self,data: pl.LazyFrame,train_tools,val_tools,test_tools,model="LSTM", input_chunk_length=40, output_chunk_length=1, hidden_dim=64, n_rnn_layers=1, batch_size=32, n_epochs=100, dropout=0.5,filepath="",meta_cols=[], target_column: str = "y",learning_rate=1e-4,base_image_path=""):
        super().__init__(data,target_column,"RNN - LSTM")
        self.loss_history = LossHistory()
        self.eary_stopping = EarlyStopping(
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
            "model_type": model,
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_dim": hidden_dim,
            "n_rnn_layers": n_rnn_layers,
            "batch_size": batch_size,
            "dropout": dropout,
            "n_epochs": n_epochs
        }
        
        # Initialization of the LSTM model using the class 'BlockRNNModel' from darts , see https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html#darts.models.forecasting.block_rnn_model.BlockRNNModel
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
                "callbacks":[self.loss_history, self.eary_stopping]     # using callbacks for the loss history/making graph and the early stopping
            },
            optimizer_kwargs={"lr":learning_rate}
        )
        self.meta_cols = self.meta_cols
        self.scaler_x_past = Scaler()
        self.scaler_x_future = Scaler()
        self.scaler_y = Scaler()
        
        # metrics
        self.res_mse = 0.0
        self.res_rmse = 0.0
        self.res_mae = 0.0
        self.res_r2 = 0.0
        self.fig_loss = None
        self.fig_pred = None
        self.Y_test_raw = None
        self.pred_df = None

    @staticmethod
    def get_formated_datetime():
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    
    '''
    Function used to optimize and perform an optimized gridsearch of hyperparameters using optuna , see https://unit8co.github.io/darts/userguide/hyperparameter_optimization.html and https://optuna.org/
    '''
    def optimize_parameters(self, n_trials: int = 20, n_epochs_optuna: int = 30):
        try:
            self.data.collect_schema()
        except:
            pass
        
        has_exo = any(c.startswith("next_") for c in self.data.columns) # if data has exogenous variables therefore darts supports the separation of both
        if has_exo:
            X_past_train_raw, X_future_train_raw, Y_train_raw = self.preprocess_to_darts_with_exogenous_variables(self.train_tools)
            X_past_val_raw, X_future_val_raw, Y_val_raw = self.preprocess_to_darts_with_exogenous_variables(self.val_tools)
        else:
            X_train_raw, Y_train_raw = self.preprocess_to_darts(self.train_tools)
            X_val_raw,   Y_val_raw   = self.preprocess_to_darts(self.val_tools)

        
        '''
            Function used by optuna at each step to perform a model training and calculate loss and scores in output
        '''
        def objective(trial):
            input_chunk_length = trial.suggest_int("input_chunk_length", 5, 60, step=5)
            hidden_dim = trial.suggest_int("hidden_dim", 8, 64, step=8)
            n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

            scaler_y = Scaler()
            Y_train = scaler_y.fit_transform(Y_train_raw)
            Y_val = scaler_y.transform(Y_val_raw)

            trial_model = BlockRNNModel(
                model=self.config_params["model_type"],
                input_chunk_length=input_chunk_length,
                output_chunk_length=1,
                hidden_dim=hidden_dim,
                n_rnn_layers=n_rnn_layers,
                batch_size=batch_size,
                dropout=dropout,
                n_epochs=n_epochs_optuna,
                pl_trainer_kwargs={"enable_progress_bar": True, "enable_model_summary": False},
                optimizer_kwargs={"lr":learning_rate}
            )
            try:
                if has_exo:
                    scaler_x_past = Scaler()
                    scaler_x_future = Scaler()
                    X_past_train = scaler_x_past.fit_transform(X_past_train_raw)
                    X_future_train = scaler_x_future.fit_transform(X_future_train_raw)
                    X_past_val = scaler_x_past.transform(X_past_val_raw)
                    X_future_val = scaler_x_future.transform(X_future_val_raw)
                    
                    trial_model.fit(
                        series=Y_train,
                        past_covariates=X_past_train,
                        future_covariates=X_future_train,
                        val_series=Y_val,
                        val_past_covariates=X_past_val,
                        val_future_covariates=X_future_val
                    )

                    pred_scaled = trial_model.historical_forecasts(
                        series=Y_val,
                        past_covariates=X_past_val,
                        future_covariates=X_future_val,
                        start=input_chunk_length,
                        forecast_horizon=1,
                        retrain=False,
                        last_points_only=True
                    )
                else:
                    scaler_x_past = Scaler()
                    X_train = scaler_x_past.fit_transform(X_train_raw)
                    X_val = scaler_x_past.transform(X_val_raw)
                    
                    trial_model.fit(
                        series=Y_train,
                        past_covariates=X_train,
                        val_series=Y_val,
                        val_past_covariates=X_val
                    )

                    pred_scaled = trial_model.historical_forecasts(
                        series=Y_val,
                        past_covariates=X_val,
                        start=input_chunk_length,
                        forecast_horizon=1,
                        retrain=False,
                        last_points_only=True
                    )

                pred = scaler_y.inverse_transform(pred_scaled)
                val_ref = Y_val_raw[input_chunk_length:]
                score = rmse(val_ref, pred)

            except Exception as e:
                print(f"Trial failed: {e}")
                return float("inf")

            return score # returning the score , in this case the root mean square error
        
        
        study = optuna.create_study(         # creating an study that's objective is too minimize the rmse error at each step
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=n_trials)
        
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            print("No trials has been done.")
            return {}
        
        best = study.best_params
        print(f"best hyperparameters : {best}")

        for k, v in best.items():
            self.config_params[k] = v
            
            
        self.loss_history = LossHistory() 
        self.eary_stopping = EarlyStopping(mode="min", patience=10, monitor="val_loss")

        self.model = BlockRNNModel(
        model=self.config_params["model_type"],
        input_chunk_length=self.config_params["input_chunk_length"],
        output_chunk_length=1,
        hidden_dim=self.config_params["hidden_dim"],
        n_rnn_layers=self.config_params["n_rnn_layers"],
        batch_size=self.config_params["batch_size"],
        dropout=self.config_params["dropout"],
        n_epochs=self.config_params["n_epochs"],
        pl_trainer_kwargs={
            "callbacks": [self.loss_history, self.eary_stopping]
            }
        )
        
        return best

    def train(self):
        has_exo = any(c.startswith("next_") for c in self.data.columns)
        if has_exo:
            X_past_train_raw, X_future_train_raw, Y_train_raw = self.preprocess_to_darts_with_exogenous_variables(self.train_tools)
            X_past_val_raw, X_future_val_raw, Y_val_raw = self.preprocess_to_darts_with_exogenous_variables(self.val_tools)
            
            X_past_train = self.scaler_x_past.fit_transform(X_past_train_raw)
            X_future_train = self.scaler_x_future.fit_transform(X_future_train_raw)
            Y_train = self.scaler_y.fit_transform(Y_train_raw)
            X_past_val = self.scaler_x_past.transform(X_past_val_raw)
            X_future_val = self.scaler_x_future.transform(X_future_val_raw)
            Y_val   = self.scaler_y.transform(Y_val_raw)

            self.model.fit(
                series=Y_train,
                past_covariates=X_past_train,
                future_covariates=X_future_train,
                val_series=Y_val,
                val_past_covariates=X_past_val,
                val_future_covariates=X_future_val
            )
        else:
            X_train_raw, Y_train_raw = self.preprocess_to_darts(self.train_tools)
            X_val_raw,   Y_val_raw   = self.preprocess_to_darts(self.val_tools)
            
            X_train = self.scaler_x_past.fit_transform(X_train_raw)
            Y_train = self.scaler_y.fit_transform(Y_train_raw)
            X_val   = self.scaler_x_past.transform(X_val_raw)
            Y_val   = self.scaler_y.transform(Y_val_raw)

            self.model.fit(
                series=Y_train,
                past_covariates=X_train,
                val_series=Y_val,
                val_past_covariates=X_val
            )

        self.fig_loss = plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history.train_losses, label="Train Loss")
        plt.plot(self.loss_history.val_losses, label="Validation Loss")
        plt.legend()

        self.test()
        return self.save()


    '''
    function used to infer on the trained (or imported model)
    
    '''
    def infer(self, X_past: TimeSeries, X_future: TimeSeries = None, n_steps: int = 1):
        has_exo = any(c.startswith("next_") for c in self.data.columns)

        X_past_scaled = self.scaler_x_past.transform(X_past)

        if has_exo:
            if X_future is None:
                raise ValueError("X_future is required because the model uses future covariates.")
            X_future_scaled = self.scaler_x_future.transform(X_future)
            
            pred_ts_scaled = self.model.predict(
                n=n_steps,
                past_covariates=X_past_scaled,
                future_covariates=X_future_scaled
            )
        else:
            pred_ts_scaled = self.model.predict(
                n=n_steps,
                past_covariates=X_past_scaled
            )

        pred_ts = self.scaler_y.inverse_transform(pred_ts_scaled)
        return pl.from_pandas(pred_ts.to_dataframe().reset_index())

    def test(self):
        has_exo = any(c.startswith("next_") for c in self.data.columns)
        if has_exo:
            X_past, X_future, Y = self.preprocess_to_darts_with_exogenous_variables(self.test_tools)
            
            X_past_scaled = self.scaler_x_past.transform(X_past)
            X_future_scaled = self.scaler_x_future.transform(X_future)
            Y_scaled = self.scaler_y.transform(Y)
            
            if len(Y_scaled) <= self.config_params["input_chunk_length"]:
                print("NaN - Not enough values to test => aborting")
                self.res_mse, self.res_rmse, self.res_mae, self.res_r2 = 0.0, 0.0, 0.0, 0.0
                self.pred_df = pl.DataFrame({"y": []})
                self.Y_test_raw = Y
                return self.pred_df
            
            pred_ts_scaled = self.model.historical_forecasts(
                series=Y_scaled,
                past_covariates=X_past_scaled,
                future_covariates=X_future_scaled,
                forecast_horizon=1,
                start=self.config_params["input_chunk_length"],
                retrain=False,
                last_points_only=True
            )
        else:
            X, Y = self.preprocess_to_darts(self.test_tools)
            
            X_scaled = self.scaler_x_past.transform(X)
            Y_scaled = self.scaler_y.transform(Y)
            
            pred_ts_scaled = self.model.historical_forecasts(
                series=Y_scaled,
                past_covariates=X_scaled,
                forecast_horizon=1,
                start=self.config_params["input_chunk_length"],
                retrain=False,
                last_points_only=True
            )
        
        pred_ts = self.scaler_y.inverse_transform(pred_ts_scaled)
        self.pred_df = pl.from_pandas(pred_ts.to_dataframe().reset_index())
        self.Y_test_raw = Y[self.config_params["input_chunk_length"]:]
        
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

        
    def save(self):
        run_path = self._save()
        return run_path
    def _save(self) -> Path:
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
                "train_tools": self.train_tools,
                "val_percentage":self.val_tools,
                "test_percentage":self.test_tools
            },
            "columns":{
                "features":self.features_cols,
                "meta":self.meta_cols,
                "target":self.target_col
            },
            "filepath":self.filepath,
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
    
    
    '''
    Convert the filtered Polars dataframe into Darts time series.
    Separate past covariates, future covariates and the target column.
    '''
    
    def preprocess_to_darts_with_exogenous_variables(self, tool_ids: list = None):
        if tool_ids is not None:
            subset = self.data.filter(pl.col("ToolIdx").is_in(tool_ids))
        else:
            subset = self.data
        df = subset.collect().sort("timestamp").drop_nulls()
        
        df_pd = df.to_pandas().reset_index(drop=True)
        future_cols = [c for c in df_pd.columns if c.startswith("next_")]
        past_cols = [c for c in df_pd.columns if c not in self.meta_cols and c not in future_cols]
        self.features_cols = past_cols + future_cols
        self.target_col = self.target_col
        X_past = TimeSeries.from_dataframe(df_pd, value_cols=past_cols)
        X_future = TimeSeries.from_dataframe(df_pd, value_cols=future_cols)
        Y = TimeSeries.from_dataframe(df_pd, value_cols=self.target_col)
        return X_past, X_future, Y
    
    
    
    '''
    Convert the filtered Polars dataframe into Darts time series.
    Separate past covariates and the target column.
    '''
    def preprocess_to_darts(self, tool_ids: list = None):
        if tool_ids is not None:
            subset = self.data.filter(pl.col("ToolIdx").is_in(tool_ids))
        else:
            subset = self.data
        df = subset.collect().sort("timestamp").drop_nulls()
        df_pd = df.to_pandas().reset_index(drop=True)
        feature_cols = [c for c in df_pd.columns if c not in self.meta_cols]
        self.features_cols = feature_cols
        self.target_col = self.target_col
        X = TimeSeries.from_dataframe(df_pd, value_cols=feature_cols)
        Y = TimeSeries.from_dataframe(df_pd, value_cols=self.target_col)
        return X, Y

    def load(self, path: Path) -> None:
        model_file = path / f"{self.model_name}.pt"
        if model_file.exists():
            self.model = BlockRNNModel.load(model_file)
        else:
            raise FileNotFoundError(f"No checkpoint found at: {model_file}")