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