import json
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.trial import TrialState



class PyTorchLSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

class PyTorchLSTM:
    def __init__(self, input_chunk_length=15, hidden_dim=16, n_rnn_layers=1, batch_size=64, n_epochs=400, dropout=0.3):
        self.model_name = "Pure_PyTorch_LSTM"
        self.base_images_path = "images/v2/models/current"
        
        self.config_params = {
            "model_type": "LSTM",
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": 1,
            "hidden_dim": hidden_dim,
            "n_rnn_layers": n_rnn_layers,
            "batch_size": batch_size,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "data_pass_type":[],
            "entry_columns":[]
        }
        
        self.meta_cols = [
            "sensor_file", "timestamp", "time", "ToolIdx", "plate_id", "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", 
            "timestamp_right", "y"
        ]
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_split, self.val_split = 0.65, 0.80
        
        self.model = None
        self.res_mse = 0.0
        self.res_rmse = 0.0
        self.res_mae = 0.0
        self.res_r2 = 0.0
        self.fig_loss = None
        self.fig_pred = None
        self.pred_df = None

    def get_formated_datetime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def create_sequences(self, X_np, Y_np,lookback=None):
        X_seq, Y_seq = [], []
        if lookback == None:
            lookback = self.config_params["input_chunk_length"]
        for i in range(len(X_np) - lookback):
            X_seq.append(X_np[i : i + lookback])
            Y_seq.append(Y_np[i + lookback])
        return np.array(X_seq), np.array(Y_seq)
    

    def train(self, df_lazy: pl.LazyFrame):
        df_cleansed = df_lazy.collect().to_dummies(columns=["DB_PASSES/SELECTION_ALLIAGE"]).sort("timestamp")
        self.config_params["data_pass_type"] = df_cleansed.select("pass_type").unique()["pass_type"].to_list()
        self.config_params["entry_columns"] = [str(c) for c in df_cleansed.get_columns()]
        df_cleansed = df_cleansed.drop(["pass_type"])
        #df_cleansed = df_lazy.to_dummies(columns=["pass_type"])
        df_cleansed = df_cleansed.drop_nulls(subset=["y"]).fill_null(0.0)

        
        feature_cols = [c for c in df_cleansed.columns if c not in self.meta_cols]
        
        X_raw = df_cleansed.select(feature_cols).to_numpy()
        Y_raw = df_cleansed.select("y").to_numpy()
        
        train_end = int(len(Y_raw) * self.train_split)
        val_end = int(len(Y_raw) * self.val_split)
        
        X_train_scaled = self.scaler_x.fit_transform(X_raw[:train_end])
        Y_train_scaled = self.scaler_y.fit_transform(Y_raw[:train_end])
        
        X_val_scaled = self.scaler_x.transform(X_raw[train_end:val_end])
        Y_val_scaled = self.scaler_y.transform(Y_raw[train_end:val_end])
        
        X_train_seq, Y_train_seq = self.create_sequences(X_train_scaled, Y_train_scaled)
        X_val_seq, Y_val_seq = self.create_sequences(X_val_scaled, Y_val_scaled)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(Y_train_seq))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(Y_val_seq))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config_params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config_params["batch_size"], shuffle=False)
        
        self.model = PyTorchLSTMNetwork(
            input_dim=X_raw.shape[1],
            hidden_dim=self.config_params["hidden_dim"],
            n_layers=self.config_params["n_rnn_layers"],
            dropout=self.config_params["dropout"]
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience, patience_counter = -1, 0
        checkpoint_path = Path("best_model_tmp.pt")
        
        for epoch in range(self.config_params["n_epochs"]):
            self.model.train()
            t_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
                
            self.model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    v_loss += loss.item()
            
            t_loss /= len(train_loader)
            v_loss /= len(val_loader)
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            
            print(f"Epoch [{epoch+1:3d}/{self.config_params['n_epochs']}] Train Loss: {t_loss:.6f} — Val Loss: {v_loss:.6f}", end="")
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(" -> Best model saved!")
            else:
                if patience != -1:
                    patience_counter += 1
                    print(f" -> Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
                    
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path))
            checkpoint_path.unlink()
            
        self.fig_loss = plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        
        self.test(df_cleansed, X_raw, Y_raw)
        self.save()
        return self.save()

    def infer(self, X_matrix: np.ndarray):
        self.model.eval()
        X_scaled = self.scaler_x.transform(X_matrix)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).numpy()
        pred_raw = self.scaler_y.inverse_transform(pred_scaled)
        return pl.DataFrame({"y": pred_raw.flatten()})

    def test(self, df_cleansed, X_raw, Y_raw):
        val_end = int(len(Y_raw) * self.val_split)
        lookback = self.config_params["input_chunk_length"]
        
        X_test_scaled = self.scaler_x.transform(X_raw[val_end:])
        X_test_seq, _ = self.create_sequences(X_test_scaled, np.zeros((len(X_test_scaled), 1)))
        
        Y_test_raw = Y_raw[val_end + lookback:]
        
        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(torch.FloatTensor(X_test_seq)).numpy()
            
        pred_ts = self.scaler_y.inverse_transform(preds_scaled)
        self.pred_df = pl.DataFrame({"y": pred_ts.flatten()})
        
        self.res_mse = mean_squared_error(Y_test_raw, pred_ts)
        self.res_rmse = np.sqrt(self.res_mse)
        self.res_mae = mean_absolute_error(Y_test_raw, pred_ts)
        self.res_r2 = r2_score(Y_test_raw, pred_ts)
        
        self.fig_pred, ax = plt.subplots(figsize=(12, 7))
        ax.plot(Y_test_raw.flatten(), label="Vrai couple", color="black", linewidth=1.5)
        ax.plot(pred_ts.flatten(), label="Couple prédit (t+1)", color="orange", linestyle="--", linewidth=1.5)
        
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
            f"Hidden Dim : {self.config_params['hidden_dim']}\n"
            f"Layers     : {self.config_params['n_rnn_layers']}\n"
            f"Lookback   : {self.config_params['input_chunk_length']} passes\n"
            f"Dropout    : {self.config_params['dropout']}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.97, 0.96, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')
        plt.tight_layout()
        return self.pred_df

    def save(self):
        run_name = f"run_{self.get_formated_datetime()}_{self.config_params['model_type']}"
        run_path = Path(self.base_images_path) / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        if self.fig_loss is not None:
            self.fig_loss.savefig(run_path / "loss_curve.png", dpi=150)
            plt.close(self.fig_loss)
        
        if self.fig_pred is not None:
            self.fig_pred.savefig(run_path / "predictions_test.png", dpi=150)
            plt.close(self.fig_pred)
        plt.close('all')

        torch.save(self.model.state_dict(), run_path / f"{self.model_name}.pt")

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
        
        with open(run_path / "metrics_config.json", "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        return run_path

    def load(self, path: Path) -> None:
        model_file = path / f"{self.model_name}.pt"
        if model_file.exists():
            self.model.load_state_dict(torch.load(model_file))
            self.model.eval()
        else:
            raise FileNotFoundError(f"No checkpoint found at: {model_file}")
        
        
    def optimize_parameters(self,df_lazy: pl.LazyFrame, n_trials=20,epoch=30):
        df_cleansed = df_lazy.collect().to_dummies(columns=["DB_PASSES/SELECTION_ALLIAGE"]).sort("timestamp")
        df_cleansed = df_cleansed.drop(["pass_type"]).drop_nulls(subset=["y"]).fill_null(0.0)
        
        feature_cols = [c for c in df_cleansed.columns if c not in self.meta_cols]
        X_raw = df_cleansed.select(feature_cols).to_numpy()
        Y_raw = df_cleansed.select("y").to_numpy()
        
        train_end = int(len(Y_raw) * self.train_split)
        val_end = int(len(Y_raw) * self.val_split)
        self.epoch = epoch
        
        X_train_scaled = self.scaler_x.fit_transform(X_raw[:train_end])
        Y_train_scaled = self.scaler_y.fit_transform(Y_raw[:train_end])
        X_val_scaled = self.scaler_x.transform(X_raw[train_end:val_end])
        Y_val_scaled = self.scaler_y.transform(Y_raw[train_end:val_end])
        
        def objective(trial):
            lookback = trial.suggest_int("input_chunk_length", 1, 30, step=2)
            hidden_dim = trial.suggest_int("hidden_dim", 8, 48, step=8)
            n_layers = trial.suggest_int("n_rnn_layers", 1, 2)
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16,32,64])
            
            X_tr_seq, Y_tr_seq = self.create_sequences(X_train_scaled, Y_train_scaled, lookback=lookback)
            X_va_seq, Y_va_seq = self.create_sequences(X_val_scaled, Y_val_scaled, lookback=lookback)
            
            tr_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(Y_tr_seq)), batch_size=batch_size, shuffle=False)
            va_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va_seq), torch.FloatTensor(Y_va_seq)), batch_size=batch_size, shuffle=False)
            
            trial_model = PyTorchLSTMNetwork(X_raw.shape[1], hidden_dim, n_layers, dropout)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(trial_model.parameters(), lr=0.001, weight_decay=weight_decay)
            
            for epoch in range(self.epoch):
                trial_model.train()
                for bx, by in tr_loader:
                    optimizer.zero_grad()
                    loss = criterion(trial_model(bx), by)
                    loss.backward()
                    optimizer.step()
            
            trial_model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    v_loss += criterion(trial_model(bx), by).item()
            return v_loss / len(va_loader)
        
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        
        for param_name, param_value in study.best_params.items():
            self.config_params[param_name] = param_value
            
        return study.best_params
            
        