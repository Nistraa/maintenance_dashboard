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
            return 
        
        def backward_fill_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
        
        def imputate_mean_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
        
        def imputate_median_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
        
        def imputate_mode_for_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
                
        def interpolate_linear_missing_data(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
        
        def interpolate_knn_missing_data(self, data: pd.DataFrame|pd.Series, neighbors: int = 5) -> pd.DataFrame|pd.Series:
            return 
        
    class HandleCategoricalVariables:
        def __init__(self) -> None:
            pass

        def label_encode_variables(self, data: pd.Series) -> pd.Series:
            return 
        
        def one_hot_encode_variables(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
        
        def target_encode_variables(self, data: pd.Series) -> pd.Series:
            return 
        

    class FeatureScaleData:
        def __init__(self) -> None:
            pass

        def min_max_scale_features(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
    
        def robust_scale_features(self, data: pd.DataFrame|pd.Series) -> pd.DataFrame|pd.Series:
            return 
