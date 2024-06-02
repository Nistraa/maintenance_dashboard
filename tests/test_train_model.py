import unittest
import numpy as np
import pandas as pd
from services import TrainModelService
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
# Test class for TrainModelService
class TestTrainModelService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        # This method will run before each test
        self.X, self.y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
    def test_train_test_split(self):
        self.fail("Not yet implemented")
        self.assertEqual(X_train.shape[0], 800)
        self.assertEqual(X_test.shape[0], 200)
        self.assertEqual(y_train.shape[0], 800)
        self.assertEqual(y_test.shape[0], 200)
        
    def test_logistic_regression(self):
        self.fail("Not yet implemented")
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        train = round(model.score(X_train, y_train) * 100, 2) 
        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        
        self.assertGreater(train, 0.7)
        self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()