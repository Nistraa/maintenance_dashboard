from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
Service to instanciate, train and test machine learning models
'''
class ModelService:
    def __init__(self, samples, features, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            samples,
            features,
            test_size=test_size,
            random_state=random_state
        )
        self.machine_learning_models = self.MachineLearningModels(self.X_train, self.X_test, self.y_train, self.y_test)


    '''
    Class containing methods to instanciate, train and test machine learning models
    '''
    class MachineLearningModels:
        def __init__(self, X_train, X_test, y_train, y_test) -> None:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.ml_models = {
                'log_reg': LogisticRegression(),
                'decision_tree': DecisionTreeClassifier(),
                #'random_forest': RandomForestClassifier(),
                'svc': SVC(),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier(),
                'xgb': XGBClassifier(),
            }


        def train_ml_model(self, model_name: str):
            model = self.ml_models.get(model_name)
            if model:
                return model.fit(self.X_train, self.y_train)
            else:
                raise ValueError(f"Model {model_name} not implemented.")
            
        def predict_ml_model(self, model_name: str):
            model = self.ml_models.get(model_name)
            if model:
                return model.predict(self.X_test)
            else:
                raise ValueError(f"Model {model_name} not implemented.")
            
        def score_ml_model(self, model_name: str):
            model = self.ml_models.get(model_name)
            if model:
                return model.score(self.X_train, self.y_train)
            else:
                raise ValueError(f"Model {model_name} not implemented.")
            
        def accuracy_score_ml_model(self, model_name: str):
            model = self.ml_models.get(model_name)
            if model:
                return accuracy_score(self.predict_ml_model(model_name), self.y_test)
            else:
                raise ValueError(f"Model {model_name} not implemented.")
            
