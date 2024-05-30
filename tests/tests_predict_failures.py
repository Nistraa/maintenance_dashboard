import unittest
import numpy as np
import pandas as pd
from services import PredictFailuresService
from tests.mock_model import MockModel

# Test class for PredictFailuresService
class TestPredictFailuresService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        self.predict_failures_service = PredictFailuresService()

    # Test method using a mock model to assure predictions
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

    