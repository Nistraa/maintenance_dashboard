import pandas as pd

# Service to preprocess data
class PreprocessDataService:
    def __init__(self):
        pass

    # Method to fill missing values in a pandas dataframe
    def preprocess_data(self, data: pd.DataFrame):
        return data.ffill()
