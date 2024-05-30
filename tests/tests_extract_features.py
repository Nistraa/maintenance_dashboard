import unittest
import numpy as np
import pandas as pd
from services import ExtractFeaturesService

# Test class for ExtractFeaturesService
class TestExtractFeaturesService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        self.extract_features_service = ExtractFeaturesService()

    # Test method using a dataframe to assure extracted features
    def test_extract_features(self):
        data = pd.DataFrame({
            'sensor_value': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]
        })

        features = self.extract_features_service.extract_features(data)
        self.assertIsNotNone(features)
        self.assertIn('sensor_mean', features)
        self.assertIn('sensor_std', features)

if __name__ == '__main__':
    unittest.main()