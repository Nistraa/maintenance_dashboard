from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Service to train ML-models
class TrainModelService:
    def __init__(self):
        pass

    # Method to train RandomForestClassifier model with a training dataset
    def train_model(self, features, labels):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        return model