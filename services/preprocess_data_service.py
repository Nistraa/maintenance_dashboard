import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from category_encoders import TargetEncoder, OrdinalEncoder
import enum



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
        Initializes the HandleMissingNumericals instance and registers the methods.
        '''
        def __init__(self) -> None:
            self.__missing_numericals_methods = {}
            self.____register_methods()

        '''
        Decorator method for assigning a name to a function in the __missing_numericals_methods dictionary
        '''
        def __methods(self, name):
            def decorator(func):
                self.__missing_numericals_methods[name] = func
                return func
            return decorator
        """
        Registers the available methods for handling missing values.
        """
        def ____register_methods(self):
            self.ffill = self.__methods('ffill')(self.ffill)
            self.bfill = self.__methods('bfill')(self.bfill)
            self.mean = self.__methods('mean')(self.mean)
            self.median = self.__methods('median')(self.median)
            self.mode = self.__methods('mode')(self.mode)
            self.interpolate_linear = self.__methods('linear')(self.interpolate_linear)
            self.interpolate_knn = self.__methods('knn')(self.interpolate_knn)

        def ffill(self, df: pd.DataFrame, *args, **kwargs):
            return df.ffill(*args, **kwargs)

        def bfill(self, df: pd.DataFrame, *args, **kwargs):
            return df.bfill(*args, **kwargs)

        def mean(self, df: pd.DataFrame, *args, **kwargs):
            return df.fillna(df.mean(*args, **kwargs))

        def median(self, df: pd.DataFrame, *args, **kwargs):
            return df.fillna(df.median(*args, **kwargs))

        def mode(self, df: pd.DataFrame, *args, **kwargs):
            return df.fillna(df.mode(*args, **kwargs).iloc[0])

        def interpolate_linear(self, df: pd.Series, *args, **kwargs):
            return df.interpolate(*args, **kwargs)

        # Refactor
        def interpolate_knn(self, df: pd.DataFrame, *args, **kwargs):
            return pd.Series([data_point[0] for data_point in KNNImputer(n_neighbors=5).fit_transform(df)])

        '''
        Method for selecting the demanded interpolation or filling method
        '''
        def select_method(self, method_name: str, df: pd.DataFrame, *args, **kwargs):
            method = self.__missing_numericals_methods.get(method_name)
            if not method:
                raise ValueError(f"Method {method_name} not recognized.")
            return(method(df, *args, **kwargs))

    class EncodeCategoricalVariables:
        '''
        Class to handle various encoding methods for categorical variables.
        '''
        # Think of possibility to register Encoders and Scaler similiar to functions.
        # Maybe subdivide Preprocess Data Service into one class for registering methods
        # and another for registering Encoders, Scalers etc.
        def __init__(self) -> None:
            '''
            Initializes the EncodeCategoricalVariables instance and registers the methods and available encoders.
            '''
            self.__encoding_methods = {}
            self.__register_methods()
            self.__encoders = {
                'label': LabelEncoder(),
                'target': TargetEncoder(),
                'ordinal': OrdinalEncoder(),
                'one_hot': self.__one_hot_encode
            }
        '''
        Decorator method for assigning a name to a function in the __encoding_methods dictionary
        '''
        def __methods(self, name: str):
            def decorator(func):
                self.__encoding_methods[name] = func
                return func
            return decorator
        """
        Registers the available methods for encoding categorical variables.
        """
        def __register_methods(self):
            self.fit_transform = self.__methods('fit_transform')(self.fit_transform)
            self.transform = self.__methods('transform')(self.transform)
            self.fit = self.__methods('fit')(self.fit)


        def fit_transform(self, encoder, *args, **kwargs):
            return encoder.fit_transform(*args,**kwargs)

        def transform(self, encoder, *args, **kwargs):
            return encoder.transform(*args,**kwargs)

        def fit(self, encoder, *args, **kwargs):
            return encoder.fit(*args,**kwargs)

        def __one_hot_encode(self, data: pd.Series) -> pd.DataFrame:
             return pd.get_dummies(data, data.name)

        '''
        Method for selecting the demanded encoder
        '''
        def select_encoder_and_method(self, encoder_name: str, encoding_method_name: str, *args, **kwargs):
            encoder = self.__encoders.get(encoder_name)
            if not encoder:
                raise ValueError(f"Method {encoder_name} not recognized.")
            if encoder_name == 'one_hot':
                return encoder(*args, **kwargs)
            encoding_method = self.__encoding_methods.get(encoding_method_name)
            if not encoding_method:
                raise ValueError(f"Method {encoding_method_name} not recognized.")
            
            return encoding_method(encoder, *args, **kwargs)


    class FeatureScaleData:
        '''
        Private dictionary containg all feature scaling __methods
        '''
        def __init__(self) -> None:
            '''
            Initializes the FeatureScaleData instance and registers the methods and available scalers.
            '''
            self.__feature_scaling___methods = {
                'min_max': MinMaxScaler().fit_transform,
                'robust': RobustScaler().fit_transform
            }

        '''
        Method for selecting the demanded scaling model
        '''
        def scale_data(self, df: pd.DataFrame, scaling_method: str):
            if scaling_method not in self.__feature_scaling___methods:
                raise ValueError(f"Method {scaling_method} not recognized.")

            return self.__feature_scaling___methods[scaling_method](df)
    class ColumnOperations:
        '''
        Class to handle various encoding methods for column operations.
        '''
        def __init__(self) -> None:
            '''
            Initializes the ColumnOperations instance and registers the methods and available arithemtic operations.
            '''
            self.__arithmetic_operation_methods = {
                'addition': lambda x, y: x + y,
                'subtraction': lambda x, y: x - y,
                'multiplication': lambda x, y: x * y,
                'division': lambda x, y: x / y,
                'square': lambda x, y: x ** x,
            }
            self.__column_operation_methods = {}
            self.__register_methods()
        '''
        Decorator method for assigning a name to a function in the __register_methods dictionary
        '''
        def __methods(self, name: str):
            def decorator(func):
                self.__column_operation_methods[name] = func
                return func
            return decorator
        """
        Registers the available methods for column operations.
        """        
        def __register_methods(self):
            self.drop_columns = self.__methods('drop')(self.drop_columns)
            self.rename_column = self.__methods('rename')(self.rename_column)
            self.mutate_column = self.__methods('mutate')(self.mutate_column)
            self.create_column = self.__methods('create')(self.create_column)

        '''       
        Application of arithemtic methods to two columns of a datframe. 
        Operation is determined by parameter operation_name
        '''
        def __arithemtic_operation(self, operation_name: str, operand_column_1: pd.Series, operand_column_2: pd.Series|None = None):
            operation = self.__arithmetic_operation_methods.get(operation_name)
            if not operation:
                raise ValueError(f"Operation {operation_name} not recognized.")  
            return operation(operand_column_1, operand_column_2)


        def rename_column(self, df: pd.DataFrame, *args, **kwargs):
            return df.rename(*args, **kwargs)
        
        def drop_columns(self, df: pd.DataFrame, *args, **kwargs):
            return df.drop(*args, **kwargs)

        '''
        Method for mutating a column given an operand column.
        Type of arithmetic operation is determined by operation_name
        '''
        def mutate_column(self, df: pd.DataFrame, operation_name: str, target_column: str, operand_column: str|None = None):
            if operand_column == None:
                operand_column = target_column
            df_copy = df.copy()
            df_copy[target_column] = self.__arithemtic_operation(operation_name, df[target_column], df[operand_column])
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
        Method for selecting the demanded interpolation or filling method
        '''        
        def select_method(self, method_name: str, df: pd.DataFrame, *args, **kwargs):
            method = self.__column_operation_methods.get(method_name)
            if not method:
                raise ValueError(f"Method {method_name} not recognized.")
            return method(df, *args, **kwargs)
