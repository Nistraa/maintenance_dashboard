import unittest
import numpy as np
import pandas as pd
from services.extract_features import ExtractFeaturesService

class TestExtractFeaturesService(unittest.TestCase):
    def setUp(self):
        self.extract_features_service = ExtractFeaturesService()

    def test_extract_features(self):
        data = pd.DataFrame({
            'sensor_value': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]
        })

        features = self.extract_features_service.extract_features(data)
        self.assertIn('mean', features)
        self.assertIn('std', features)

if __name__ == '__main__':
    unittest.main()