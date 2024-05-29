from matplotlib import pyplot as plt
import pandas as pd



class VisualizeDataService:
    def __init__(self):
        pass

    def visualize_data(self, data: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestamp'], data['sensor_value'], label='Sensor Value')
        plt.axhline(y=100, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Value')
        plt.title('Sensor Data Over Time')
        plt.legend()
        plt.show()
        return True