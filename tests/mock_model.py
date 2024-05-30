
# Mock model for testing
class MockModel:
    def predict(self, feature):
        return [0] * len(feature)
    
