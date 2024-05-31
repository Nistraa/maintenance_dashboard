import pandas as pd


# Service to load data from file
class LoadDataService:
    def __init__(self):
        pass

    # Method to read data from given file path
    def read_data(self, file_path):
        return pd.read_csv(file_path)
    
    '''
    # TO BE DONE
    '''
    def generate_sqlmodel(self, data: pd.DataFrame):
        return NotImplementedError()