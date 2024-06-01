import unittest
from services import PreprocessDataService
import pandas as pd
import pandas.testing as pdt

'''
Test class for PreprocessDataService
'''
class TestPreprocessDataService(unittest.TestCase):
    '''
    Unittest special method to setup the test with a fresh state and test cases
    '''
    def setUp(self):
        self.preprocess_data_service = PreprocessDataService()
        self.numerical_data_falsy = pd.DataFrame({
            'sensor_value': [100, 110, None, 115, 120]
            })
        self.numerical_data = pd.DataFrame({
            'sensor_target': [5, 10, 15, 20],
            'sensor_operand': [10, 15, 20, 25],
            'sensor_to_drop_1': [0, 0, 0, 0],
            'sensor_to_drop_2': [0, 0, 0, 0]
            })
        self.categorical_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'target': [1, 0, 1, 0, 1, 0]        
            })  
        self.scaler_data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    '''
    Test method for forward filling missing data
    '''
    def test_forward_fill_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, 'ffill')
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], 110.0)

    '''
    Test method for backward filling missing data
    '''
    def test_backward_fill_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, fill_method='bfill')
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], 115.0)

    '''
    Test method for using the mean for missing data
    '''
    def test_imputate_mean_for_missing_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, 'mean', fillna=True)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numerical_data_falsy['sensor_value'].mean())

    '''
    Test method for using the median for missing data
    '''
    def test_imputate_median_for_missing_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, 'median', fillna=True)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numerical_data_falsy['sensor_value'].median())

    '''
    Test method for using the mode for missing data
    '''
    def test_imputate_mode_for_missing_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, 'mode', fillna=True)
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())
        self.assertEqual(processed_data['sensor_value'].iloc[2], self.numerical_data_falsy['sensor_value'].mode().iloc[0])

    '''
    Test method for interpolating linear values for missing data
    '''
    def test_interpolate_linear_missing_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, fill_method='linear')
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())

    '''
    Test method for using the k-nearest-neighbour heuristic for missing data
    '''
    def test_interpolate_knn_missing_data(self):
        processed_data = self.preprocess_data_service.handle_missing_numericals.fill_missing_numerical(self.numerical_data_falsy, fill_method='knn')
        self.assertIsNotNone(processed_data)
        self.assertFalse(processed_data.isnull().values.any())

    '''
    Test label encoding method
    '''
    def test_label_encode_variables(self):
        processed_data = self.preprocess_data_service.encode_categorical_variables.encode_categorical_variables(self.categorical_data['category'], 'label')
        expected_data = [0,1,2,0,1,2]
        self.assertIsNotNone(processed_data)
        self.assertListEqual(list(processed_data), expected_data)

    '''
    Test one-hot encoding method
    '''
    def test_one_hot_encode_variables(self):
        processed_data = self.preprocess_data_service.encode_categorical_variables.encode_categorical_variables([self.categorical_data['category'], 'category'], 'one_hot')
        expected_columns = ['category_A', 'category_B', 'category_C']
        self.assertListEqual(list(processed_data.columns), expected_columns)
        self.assertEqual(processed_data.shape, (6, 3))
        self.assertFalse('category' in processed_data.columns)

    '''
    Test target encoding method
    '''
    def test_target_encode_variables(self):
        processed_data = self.preprocess_data_service.encode_categorical_variables.encode_categorical_variables(self.categorical_data, 'target')
        expected_means = {
            'A': 0.5,
            'B': 0.5,
            'C': 0.5
        }
        for category in expected_means:
            self.assertAlmostEqual(processed_data['category'].iloc[0], expected_means[category])


    '''
    Test feature scaling for missing data
    '''
    def test_min_max_scale_features(self):
        processed_data = self.preprocess_data_service.feature_scale_data.scale_data(self.scaler_data, 'min_max')
        self.assertIsNotNone(processed_data)

    '''
    Test robust scaling for missing data
    '''
    def test_robust_scale_features(self):
        processed_data = self.preprocess_data_service.feature_scale_data.scale_data(self.scaler_data, 'robust')
        self.assertIsNotNone(processed_data)

    '''
    Test for functionality of dropping a column
    '''
    def test_drop_columns(self):
        manipulated_data = self.preprocess_data_service.column_operations.drop_columns(self.numerical_data, ['sensor_to_drop_1', 'sensor_to_drop_2'])

        pdt.assert_frame_equal(manipulated_data, self.numerical_data.drop(['sensor_to_drop_1', 'sensor_to_drop_2'], axis='columns'))
    '''
    Test addition of two columns
    '''
    def test_arithmetic_operations_addition(self):
        manipulated_data = self.preprocess_data_service.column_operations.mutate_column(self.numerical_data, 'addition', 'sensor_target', 'sensor_operand')
        self.numerical_data['sensor_target'] = self.numerical_data['sensor_target'] + self.numerical_data['sensor_operand']
        
        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data)

    '''
    Test subtraction of two columns
    '''
    def test_arithmetic_operations_subtraction(self):
        manipulated_data = self.preprocess_data_service.column_operations.mutate_column(self.numerical_data, 'subtraction', 'sensor_target', 'sensor_operand')
        self.numerical_data['sensor_target'] = self.numerical_data['sensor_target'] - self.numerical_data['sensor_operand']

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data)
    '''
    Test multiplication of two columns
    '''
    def test_arithmetic_operations_multiplication(self):
        manipulated_data = self.preprocess_data_service.column_operations.mutate_column(self.numerical_data, 'multiplication', 'sensor_target', 'sensor_operand')
        self.numerical_data['sensor_target'] = self.numerical_data['sensor_target'] * self.numerical_data['sensor_operand']

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data)
    '''
    Test division of two columns
    '''
    def test_arithmetic_operations_division(self):
        manipulated_data = self.preprocess_data_service.column_operations.mutate_column(self.numerical_data, 'division', 'sensor_target', 'sensor_operand')
        self.numerical_data['sensor_target'] = self.numerical_data['sensor_target'] / self.numerical_data['sensor_operand']

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data)
    '''
    Test squaring of two columns
    '''
    def test_arithmetic_operations_square(self):
        manipulated_data = self.preprocess_data_service.column_operations.mutate_column(self.numerical_data, 'square', 'sensor_target')
        self.numerical_data['sensor_target'] = self.numerical_data['sensor_target'] ** self.numerical_data['sensor_target']

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data)
    '''
    Tests for creating a new column
    '''
    def test_column_creation_square(self):
        manipulated_data = self.preprocess_data_service.column_operations.create_column(self.numerical_data, 'square', 'new_column', 'sensor_operand')

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        self.assertTrue(len(manipulated_data.columns) > len(self.numerical_data.columns))

    def test_column_creation_subtraction(self):
        manipulated_data = self.preprocess_data_service.column_operations.create_column(self.numerical_data, 'subtraction', 'new_column', 'sensor_target', 'sensor_operand')

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        self.assertTrue(len(manipulated_data.columns) > len(self.numerical_data.columns))
    
    '''
    Test for renaming of a column
    '''
    def test_column_renaming(self):
        manipulated_data = self.preprocess_data_service.column_operations.rename_column(self.numerical_data_falsy, 'sensor_value', 'changed')

        self.assertIsNotNone(manipulated_data)
        self.assertIsInstance(manipulated_data, pd.DataFrame)
        pdt.assert_frame_equal(manipulated_data, self.numerical_data_falsy.rename(columns={'sensor_value': 'changed'}))

if __name__ == '__main__':
    unittest.main()