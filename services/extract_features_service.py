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
            self.__pandas_methods = {}
            self.register_methods()

        def method(self, name: str):
            def decorator(func):
                self.__pandas_methods[name] = func
                return func
            return decorator
        
        def register_methods(self):
            self.head = self.method('head')(self.head)
            self.describe = self.method('describe')(self.describe)
            self.info = self.method('info')(self.info)
            self.isna = self.method('isna')(self.isna)
            self.uniques = self.method('uniques')(self.uniques)

            
        def head(self, df: pd.DataFrame, *args, **kwargs):
            return df.head(*args, **kwargs)
        
        def describe(self, df: pd.DataFrame, *args, **kwargs):
            return df.describe(*args, **kwargs)
        
        def isna(self, df: pd.DataFrame, *args, **kwargs):
            return df.isna(*args, **kwargs)
        
        def uniques(self, df: pd.DataFrame, *args, **kwargs):
            return df.nunique(*args, **kwargs)
        
        def info(self, df: pd.DataFrame, *args, **kwargs):
            buffer = StringIO()
            df.info(buf=buffer)
            lines = buffer.getvalue()
            file = [line.split() for line in lines.splitlines()[3:-2]]
            return pd.DataFrame(file)
        

        def select_method(self, df: pd.DataFrame, method_name: str, *args, **kwargs):
            method = self.__pandas_methods.get(method_name)
            if method:
                return method(df, *args, **kwargs)
            else:
                raise ValueError(f"Method {method_name} not recognized.")