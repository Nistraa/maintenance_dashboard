import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from category_encoders import TargetEncoder


# Service to preprocess data
class PreprocessDataService:
    def __init__(self) -> None:
        pass
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
