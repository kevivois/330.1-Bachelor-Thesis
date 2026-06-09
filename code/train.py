import polars as pl
from models.LSTMModel import LSTMModel
def train():
    
    data_path = "data/v2/tsfel_extracted_new.csv"
    data = pl.scan_csv(data_path).lazy()
    model = LSTMModel()
    model.start_training(data,"y",saving=False)

if __name__ == "__main__":
    train()