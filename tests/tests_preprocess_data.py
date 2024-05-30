import unittest
from services import PreprocessDataService
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np

'''
Test class for PreprocessDataService
'''
class TestPreprocessDataService(unittest.TestCase):
    '''
    Unittest special method to setup the test with a fresh state and test cases
    '''
    def setUp(self):
        self.preprocess_data_service = PreprocessDataService()
        self.numericalData = pd.DataFrame({
            'sensor_value': [100, 110, None, 115, 120]
            })
        self.categoricalData = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'target': [1, 0, 1, 0, 1, 0]        
            })
        self.le = LabelEncoder()
        
    '''
    Test method for forward filling missing data
    '''
    def test_forward_fill_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.forward_fill_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], 100)

    '''
    Test method for backward filling missing data
    '''
    def test_forward_fill_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.backward_fill_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], 115)

    '''
    Test method for using the mean for missing data
    '''
    def test_imputate_mean_for_missing_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.imputate_mean_for_missing_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numericalData['sensor_value'].mean())

    '''
    Test method for using the median for missing data
    '''
    def test_imputate_median_for_missing_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.imputate_median_for_missing_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numericalData['sensor_value'].mean())

    '''
    Test method for using the mode for missing data
    '''
    def test_imputate_mean_for_missing_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.imputate_mode_for_missing_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numericalData['sensor_value'].mode().iloc[0])

    '''
    Test method for interpolating linear values for missing data
    '''
    def test_interpolate_linear_missing_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.interpolate_linear_missing_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())

    '''
    Test method for using the k-nearest-neighbour heuristic for missing data
    '''
    def test_interpolate_knn_missing_data(self):
        processed_data = self.preprocess_data_service.HandleMissingNumericals.interpolate_knn_missing_data(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())

    '''
    Test label encoding method
    '''
    def test_label_encode_variables(self):
        processed_data = self.preprocess_data_service.HandleCategoricalVariables.label_encode_variables(self, self.categoricalData['category'])
        expected_data = [0,1,2,0,2]
        self.assertIsNotNone(processed_data)
        self.assertListEqual(processed_data.toList(), expected_data)

    '''
    Test label encoding method for consistency
    '''
    def test_label_encode_variables_consistency(self):
        processed_data_1 = self.preprocess_data_service.HandleCategoricalVariables.label_encode_variables(self, self.categoricalData['category'])
        processed_data_2 = self.preprocess_data_service.HandleCategoricalVariables.label_encode_variables(self, self.categoricalData['category'])
        self.assertIsNotNone(processed_data_1)
        self.assertIsNotNone(processed_data_2)
        self.assertListEqual(processed_data_1.toList(), processed_data_2.toList())

    '''
    Test one-hot encoding method
    '''
    def test_one_hot_encode_variables(self):
        processed_data = self.preprocess_data_service.HandleCategoricalVariables.one_hot_encode_variables(self, self.categoricalData['category'])
        expected_columns = ['category_A', 'category_B', 'category_C']
        self.assertListEqual(list(processed_data.columns), expected_columns)
        self.assertEqual(processed_data.shape, (6, 3))
        self.assertFalse('category' in processed_data.columns)

    '''
    Test target encoding method
    '''
    def test_target_encode_variables(self):
        processed_data = self.preprocess_data_service.HandleCategoricalVariables.target_encode_variables(self, self.categoricalData)
        expected_means = {
            'A': 0.5,
            'B': 0.5,
            'C': 0.5
        }
        for category in expected_means:
            self.assertAlmostEqual(processed_data[self.categoricalData['category'] == category].iloc[0], expected_means[category])

    '''
    Test target encoding method for consistency
    '''
    def test_target_encode_variables_consistency(self):
        processed_data_1 = self.preprocess_data_service.HandleCategoricalVariables.target_encode_variables(self, self.categoricalData)
        processed_data_2 = self.preprocess_data_service.HandleCategoricalVariables.target_encode_variables(self, self.categoricalData)
        self.assertIsNotNone(processed_data_1)
        self.assertIsNotNone(processed_data_2)
        self.assertListEqual(processed_data_1.toList(), processed_data_2.toList())

    '''
    Test feature scaling for missing data
    '''
    def test_feature_scaling(self):
        processed_data = self.preprocess_data_service.FeatureScaleData.min_max_scale_features(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())

    '''
    Test robust scaling for missing data
    '''
    def test_feature_scaling(self):
        processed_data = self.preprocess_data_service.FeatureScaleData.robust_scale_features(self, self.numericalData)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())


if __name__ == '__main__':
    unittest.main()