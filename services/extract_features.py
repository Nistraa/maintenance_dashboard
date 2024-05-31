import pandas as pd
from io import StringIO

'''
Service to extract features from data
'''
class ExtractFeaturesService:
    def __init__(self) -> None:
        self.pandas_standard_methods = self.PandasStandardMethods()

    '''
    Method to extract basic features of dataframe
    '''
    class PandasStandardMethods:
        def __init__(self) -> None:
            self._pandas_methods = {
            'head': pd.DataFrame.head,
            'describe': pd.DataFrame.describe,
            'info': self._info,
            'isna': pd.DataFrame.isna,
            'uniques': self._uniques,
        }
            
        def _info(self, df: pd.DataFrame):
            buffer = StringIO()
            df.info(buf=buffer)
            lines = buffer.getvalue()
            file = [line.split() for line in lines.splitlines()[3:-2]]
            return pd.DataFrame(file)
        
        def _uniques(self, df: pd.DataFrame):
            return df.apply(lambda x: x.nunique())

        def pandas_standard_method(self, df: pd.DataFrame, method_name: str):
            method = self._pandas_methods.get(method_name)
            if method:
                return method(df)
            else:
                raise ValueError(f"Method {method_name} not recognized.")