import unittest
import numpy as np
import pandas as pd
from services import PredictFailuresService

class MockModel:
    def predict(self, feature):
        return [0] * len(feature)
    
class TestPredictFailuresService(unittest.TestCase):
    def setUp(self):
        self.predict_failures_service = PredictFailuresService()

    def test_predict_failures(self):
        model = MockModel()
        features = pd.DataFrame({
            'mean': [120],
            'std': [7]
        })
        predictions = self.predict_failures_service.predict_failures(model, features)
        self.assertEqual(len(predictions), 1)

if __name__ == '__main__':
    unittest.main()

    