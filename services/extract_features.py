import pandas as pd


# Service to extract features from data
class ExtractFeaturesService:
    def __init__(self) -> None:
        pass

    # Method to calculate rolling mean and standard deviation of sensor values
    class PandasStandardMethods:
        def __init__(self) -> None:
            self.pandas_methods = {
        }

        def pandas_standard_method(self, df: pd.DataFrame, method_name: str):
            return NotImplementedError()
            return self.pandas_methods[method_name](df) if method_name in self.pandas_methods else ValueError(f"Method {method_name} not recognized.")