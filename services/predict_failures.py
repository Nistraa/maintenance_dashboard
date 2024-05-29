from sklearn.ensemble import RandomForestClassifier




class PredictFailuresService:
    def __init__(self):
        pass

    def predict_failures(self, model: RandomForestClassifier, features):
        return model.predict(features)