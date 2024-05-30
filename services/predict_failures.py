from sklearn.ensemble import RandomForestClassifier



# Service to predict failures using ML-models
class PredictFailuresService:
    def __init__(self):
        pass

    # Method to predict failures using RandomForestClassifier
    def predict_failures(self, model: RandomForestClassifier, features):
        return model.predict(features)