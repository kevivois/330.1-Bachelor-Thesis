import polars as pl
from models.v2.PyTorchLSTM import PyTorchLSTM
def train():
    
    data_path = "data/v4/tsfel_extracted_new.csv"
    data = pl.scan_csv(data_path).lazy()
    model = PyTorchLSTM()
    model.train(data)

if __name__ == "__main__":
    train()