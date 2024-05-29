import unittest
import numpy as np
import pandas as pd
from services import TrainModelService

class TestTrainModelService(unittest.TestCase):
    def setUp(self):
        self.train_model_service = TrainModelService()

    def test_train_model(self):
        features = pd.DataFrame({
            'sensor_mean': [110, 115, 120],
            'sensor_std': [5, 6, 4]
        })
        labels = pd.Series([0, 1, 0])
        model = self.train_model_service = TrainModelService().train_model(features, labels)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()