import unittest
from services import ModelService
from sklearn.datasets import make_classification


'''
Test class for ModelService
'''
class TestTrainModelService(unittest.TestCase):
    '''
    Unittest special method to setup the test with a fresh state
    '''
    def setUp(self):
        self.samples, self.features = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.model_service = ModelService(samples=self.samples, features=self.features, test_size=0.2, random_state=42)
        self.models = ['log_reg', 'decision_tree', 'svc', 'naive_bayes', 'knn', 'xgb']
        
    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = self.model_service.X_train, self.model_service.X_test, self.model_service.y_train, self.model_service.y_test
        self.assertEqual(X_train.shape[0], 800)
        self.assertEqual(X_test.shape[0], 200)
        self.assertEqual(y_train.shape[0], 800)
        self.assertEqual(y_test.shape[0], 200)
        
    def test_machine_learning_models(self):
        for model in self.models:
            self.model_service.machine_learning_models.train_ml_model(model_name=model)
            score = self.model_service.machine_learning_models.score_ml_model(model_name=model)
            accuracy = self.model_service.machine_learning_models.accuracy_score_ml_model(model_name=model)

            print(f'{model} score: {score}')
            print(f'{model} accuracy: {accuracy}')

            self.assertGreater(score, 0.7)
            self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()