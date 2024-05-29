import pandas as pd



class ExtractFeaturesService:
    def __init__(self):
        pass

    def extract_features(self, data: pd.DataFrame):
        data['sensor_mean'] = data['sensor_value'].rolling(window=10).mean()
        data['sensor_std'] = data['sensor_value'].rolling(window=10).std()
        data.dropna(inplace=True)
        return data