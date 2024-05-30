import pandas as pd


# Service to extract features from data
class ExtractFeaturesService:
    def __init__(self):
        pass

    # Method to calculate rolling mean and standard deviation of sensor values
    def extract_features(self, data: pd.DataFrame):
        data['sensor_mean'] = data['sensor_value'].rolling(window=10).mean()
        data['sensor_std'] = data['sensor_value'].rolling(window=10).std()
        data.dropna(inplace=True)
        return data