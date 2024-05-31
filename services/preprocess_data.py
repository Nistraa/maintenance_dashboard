import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from category_encoders import TargetEncoder
import numpy as np


# Service to preprocess data
class PreprocessDataService:
    def __init__(self) -> None:
        self.column_operations = self.ColumnOperations()
    class HandleMissingNumericals:
        def __init__(self) -> None:
            pass

        '''
        Fills missing data with precedent values from sequence
        '''
        def forward_fill_data(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.ffill()
        
        '''
        Fills missing data with antecedent values from sequence
        '''
        def backward_fill_data(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.bfill()
        
        '''
        Fills missing data with mean of sequence
        '''
        def imputate_mean_for_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.fillna(data.mean())
        '''
        Fills missing data with middle value
        '''
        def imputate_median_for_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.fillna(data.median())
        '''
        Fills missing data with most occuring value
        '''
        def imputate_mode_for_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.fillna(data.mode().iloc[0])
        '''
        Interpolates missing data by estimating within a range of data points
        '''     
        def interpolate_linear_missing_data(self, data: pd.Series) -> pd.Series:
            return data.interpolate(method='linear')
        '''
        Interpolates missing data based on its nearest neighbours
        '''
        def interpolate_knn_missing_data(self, data: pd.DataFrame, neighbors: int = 5) -> pd.Series:
            return pd.Series([data_point[0] for data_point in KNNImputer(n_neighbors=neighbors).fit_transform(data)])
        
    class HandleCategoricalVariables:
        def __init__(self) -> None:
            pass

        '''
        Converts categories to unique integers
        '''
        def label_encode_variables(self, data: pd.Series) -> pd.Series:
            return LabelEncoder().fit_transform(data)
        '''
        Converts categories to binary vectors
        '''
        def one_hot_encode_variables(self, data: pd.Series, columns) -> pd.Series:
            return pd.get_dummies(data, columns)
        '''
        Converts categories to the mean of the target variable
        '''
        def target_encode_variables(self, category: pd.Series, target: pd.Series) -> pd.DataFrame:
            return TargetEncoder().fit_transform(category, target)
        

    class FeatureScaleData:
        def __init__(self) -> None:
            pass

        '''
        Scales data within in a specifeid range
        '''
        def min_max_scale_features(self, data: pd.Series) -> pd.Series:
            return MinMaxScaler().fit_transform(data)
        '''
        Scales data robust to outliers'''
        def robust_scale_features(self, data: pd.Series) -> pd.Series:
            return RobustScaler().fit_transform(data)
        
    class ColumnOperations:
        def __init__(self) -> None:
            self._arithmetic_methods = {
                'addition': lambda x, y: x + y,
                'subtraction': lambda x, y: x - y,
                'multiplication': lambda x, y: x * y,
                'division': lambda x, y: x / y,
                'square': lambda x, y: x ** x,

            }

        def _arithemtic_operation(self, operation_name: str, operand_column_1: pd.Series, operand_column_2: pd.Series|None = None):
            method = self._arithmetic_methods.get(operation_name)
            if method:
                result = method(operand_column_1, operand_column_2)
                return result
            else:
                raise ValueError(f"Operation {operation_name} not recognized.")  
                
        def drop_columns(self, df: pd.DataFrame, columns: list[str]):
            return df.drop(columns, axis='columns')
        
        def mutate_column(self, df: pd.DataFrame, operation_name: str, target_column: str, operand_column: str|None = None):
            if operand_column == None:
                operand_column = target_column
            df_copy = df.copy()
            df_copy[target_column] = self._arithemtic_operation(operation_name, df[target_column], df[operand_column])
            return df_copy
        
        def create_column(self, df: pd.DataFrame, operation_name: str, new_column: str, operand_column_1: str, operand_column_2: str|None = None):
            if operand_column_2 == None:
                operand_column_2 = operand_column_1
            df_copy = df.copy()
            df_copy[new_column] = self._arithemtic_operation(operation_name, df[operand_column_1], df[operand_column_2])
            return df_copy
        
        def rename_column(self, df: pd.DataFrame, column_name: str):
            return df.rename(column_name)