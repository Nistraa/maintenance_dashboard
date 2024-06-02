from services import LoadDataService, PreprocessDataService, ExtractFeaturesService, ModelService
from sklearn.metrics import classification_report, confusion_matrix

'''
Initialze modules
'''
load_data = LoadDataService()
preprocess_data = PreprocessDataService()
extract_features = ExtractFeaturesService()

'''
Read data and create dataframe
'''
dataframe = load_data.read_data('dataset/predictive_maintenance.csv')

'''
Drop columns not needed
'''
dataframe = preprocess_data.column_operations.drop_columns(dataframe,['UDI', 'Product ID'])

'''
Create a new column depciting the temperature differnce
'''
dataframe = preprocess_data.column_operations.create_column(
    df=dataframe,
    operation_name='subtraction',
    new_column='temperature difference [K]',
    operand_column_1="Process temperature [K]",
    operand_column_2="Air temperature [K]"
)

'''
Encode categorial variables in the dataframe
'''
dataframe['Type'] = preprocess_data.encode_categorical_variables.encode_categorical_variables(df=dataframe, encoding_method='ordinal', category_column='Type')
dataframe['Failure Type'] = preprocess_data.encode_categorical_variables.encode_categorical_variables(dataframe, encoding_method='ordinal', category_column='Failure Type')
dataframe['Failure Type'] = preprocess_data.encode_categorical_variables.encode_categorical_variables(dataframe, encoding_method='label', category_column='Failure Type')


'''
Make classification
'''
X = preprocess_data.column_operations.drop_columns(dataframe, ['Failure Type'])
y = dataframe['Failure Type']
model_service = ModelService(samples=X, features=y, test_size=0.2, random_state=21)

'''
Train and test model for accuracy and score.
Currently available models:

Logistic Regression: 'log_reg'
Decision Tree Classifier: 'decision_tree'
Support Vector Machines: 'svc'
Naive Bayes: 'naive_bayes'
KNeighbors Classifier: 'knn'

'''

models_selected = ['log_reg', 'svc', 'knn']

for model in models_selected:
    model_service.machine_learning_models.train_ml_model(model)
    model_score = model_service.machine_learning_models.score_ml_model(model)
    model_predictions = model_service.machine_learning_models.predict_ml_model(model)
    model_accuracy = model_service.machine_learning_models.accuracy_score_ml_model(model)

    print(f'{model} training accuracy: {model_score}')
    print(f'{model} testing accuracy: {model_accuracy}')
    print("\033[1m--------------------------------------------------------\033[0m")
    print(f'{model} classification report:  \n {classification_report(model_service.y_test, model_predictions, zero_division=0.0)}')
    print("\033[1m--------------------------------------------------------\033[0m")
    print(f'{model} confusion matrix: \n {confusion_matrix(model_service.y_test, model_predictions)}')
    print("\033[1m--------------------------------------------------------\033[0m")
    print("\033[1m--------------------------------------------------------\033[0m")
