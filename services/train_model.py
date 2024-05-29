from sklearn.ensemble import RandomForestClassifier



class TrainModelService:
    def __init__(self):
        pass

    def train_model(self, features, labels):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        return model