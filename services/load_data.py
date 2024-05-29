import pandas as pd


class LoadDataService:
    def __init__(self):
        pass

    def read_data(self, file_path):
        return pd.read_csv(file_path)