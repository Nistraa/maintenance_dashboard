import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from category_encoders import TargetEncoder, OrdinalEncoder


'''
Service to preprocess data
'''
class PreprocessDataService:
    def __init__(self) -> None:
        self.column_operations = self.ColumnOperations()
        self.handle_missing_numericals = self.HandleMissingNumericals()
        self.encode_categorical_variables = self.EncodeCategoricalVariables()
        self.feature_scale_data = self.FeatureScaleData()

    class HandleMissingNumericals:
        '''
        Private dictionary containg all interpolation and filling methods
        '''
        def __init__(self) -> None:
            self.__missing_numericals_methods = {
                'ffill': pd.DataFrame.ffill,
                'bfill': pd.DataFrame.bfill,
                'mean': pd.DataFrame.mean,
                'median': pd.DataFrame.median,
                'mode': self.__mode,
                'linear': self.__interpolate_linear,
                'knn': self.__interpolate_knn
            }

        def __mode(self, df: pd.DataFrame):
            return df.mode().iloc[0]
        
        def __interpolate_linear(self, data: pd.Series) -> pd.Series:
            return data.interpolate(method='linear')

        def __interpolate_knn(self, data: pd.DataFrame, neighbors: int = 5) -> pd.Series:
            return pd.Series([data_point[0] for data_point in KNNImputer(n_neighbors=neighbors).fit_transform(data)])

        '''
        Method for selecting the demanded interpolation or filling method
        '''
        def fill_missing_numerical(self, df: pd.DataFrame, fill_method: str, fillna: bool = False):
            if fill_method not in self.__missing_numericals_methods:
                raise ValueError(f"Method {fill_method} not recognized.")
            
            return df.fillna(self.__missing_numericals_methods[fill_method](df)) if fillna else self.__missing_numericals_methods[fill_method](df)
        
    class EncodeCategoricalVariables:
        '''
        Class to handle various encoding methods for categorical variables.
        '''
        def __init__(self) -> None:
            self.__encoding_method = {
                'label': self.__label_encode,
                'one_hot': self.__one_hot_encode,
                'target': self.__target_encode,
                'ordinal': self.__ordinal_encode,
            }


        def __label_encode(self, data: pd.Series) -> pd.Series:
            return LabelEncoder().fit_transform(data)

        def __one_hot_encode(self, data: pd.Series) -> pd.DataFrame:
            return pd.get_dummies(data, data.name)

        def __target_encode(self, df: pd.DataFrame, category_column: str, target_column: str) -> pd.DataFrame:
            return TargetEncoder().fit_transform(df[category_column], df[target_column])

        def __ordinal_encode(self, df: pd.Series) -> pd.Series:
            return OrdinalEncoder().fit_transform(df)

        '''
        Method for selecting the demanded encoder
        '''
        def encode_categorical_variables(self, df: pd.DataFrame, encoding_method: str, category_column, target_column: str = None):
            if encoding_method not in self.__encoding_method:
                raise ValueError(f"Method {encoding_method} not recognized.")
            
            if encoding_method == 'target':
                if not target_column:
                    raise ValueError("Target column must be provided for target encoding.")
                return self.__encoding_method[encoding_method](df, category_column, target_column)
            return self.__encoding_method[encoding_method](df[category_column])


    class FeatureScaleData:
        '''
        Private dictionary containg all feature scaling methods
        '''
        def __init__(self) -> None:
            self.__feature_scaling_methods = {
                'min_max': MinMaxScaler().fit_transform,
                'robust': RobustScaler().fit_transform
            }

        '''
        Method for selecting the demanded scaling model
        '''
        def scale_data(self, df: pd.DataFrame, scaling_method: str):
            if scaling_method not in self.__feature_scaling_methods:
                raise ValueError(f"Method {scaling_method} not recognized.")

            return self.__feature_scaling_methods[scaling_method](df)
    class ColumnOperations:
        '''
        Private dictionary containg all arithemtic methods
        '''
        def __init__(self) -> None:
            self.__arithmetic_methods = {
                'addition': lambda x, y: x + y,
                'subtraction': lambda x, y: x - y,
                'multiplication': lambda x, y: x * y,
                'division': lambda x, y: x / y,
                'square': lambda x, y: x ** x,

            }

        '''
        Application of arithemtic methods to two columns of a datframe. 
        Operation is determined by parameter operation_name
        '''
        def __arithemtic_operation(self, operation_name: str, operand_column_1: pd.Series, operand_column_2: pd.Series|None = None):
            if operation_name not in self.__arithmetic_methods:
                raise ValueError(f"Operation {operation_name} not recognized.")  
            
            return self.__arithmetic_methods[operation_name](operand_column_1, operand_column_2)
            
        '''

        Method for mutating a column given an operand column.
        Type of arithmetic operation is determined by operation_name
        '''
        def mutate_column(self, df: pd.DataFrame, operation_name: str, target_columnumn: str, operand_column: str|None = None):
            if operand_column == None:
                operand_column = target_columnumn
            df_copy = df.copy()
            df_copy[target_columnumn] = self.__arithemtic_operation(operation_name, df[target_columnumn], df[operand_column])
            return df_copy
        
        '''
        Method for creating a new column based on existing data in 
        dataframe
        '''
        def create_column(self, df: pd.DataFrame, operation_name: str, new_column: str, operand_column_1: str, operand_column_2: str|None = None):
            if operand_column_2 == None:
                operand_column_2 = operand_column_1
            df_copy = df.copy()
            df_copy[new_column] = self.__arithemtic_operation(operation_name, df[operand_column_1], df[operand_column_2])
            return df_copy
        
        '''
        Method for renaming column'''
        def rename_column(self, df: pd.DataFrame, old_column_name: str, new_column_name):
            return df.rename(columns={old_column_name: new_column_name})
        
        '''
        Method for deleting columns
        '''        
        def drop_columns(self, df: pd.DataFrame, columns: list[str]):
            return df.drop(columns, axis='columns')
