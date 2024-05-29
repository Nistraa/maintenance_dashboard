import pandas as pd


class PreprocessDataService:
    def __init__(self):
        pass

    def preprocess_data(self, data: pd.DataFrame):
        return data.ffill()
