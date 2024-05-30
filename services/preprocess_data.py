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

        def forward_fill_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.ffill()
        
        def backward_fill_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.bfill()
        
        def imputate_mean_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.fillna(data.mean())
        
        def imputate_median_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.fillna(data.median())
        
        def imputate_mode_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.fillna(data.mode().iloc[0])
                
        def interpolate_linear_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return data.interpolate(method='linear')
        
        def interpolate_knn_missing_data(self, data: pd.DataFrame|pd.Series, neighbors: int = 5) -> pd.Series:
            return pd.Series([data_point[0] for data_point in KNNImputer(n_neighbors=neighbors).fit_transform(data)])
        
    class HandleCategoricalVariables:
        def __init__(self) -> None:
            pass

        def label_encode_variables(self, data: pd.Series) -> pd.Series:
            return LabelEncoder().fit_transform(data)
        
        def one_hot_encode_variables(self, data: pd.DataFrame|pd.Series, columns) -> pd.DataFrame|pd.Series:
            return pd.get_dummies(data, columns)
        
        def target_encode_variables(self, category: pd.Series, target: pd.Series) -> pd.DataFrame:
            return TargetEncoder().fit_transform(category, target)
        

    class FeatureScaleData:
        def __init__(self) -> None:
            pass

        def min_max_scale_features(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return MinMaxScaler().fit_transform(data)
    
        def robust_scale_features(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return RobustScaler().fit_transform(data)
