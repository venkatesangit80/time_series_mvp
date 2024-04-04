from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from module.RNN import RNN
import pandas as pd
from datetime import datetime
import numpy as np

class PyTorchGrid(BaseEstimator):
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def fit(self, X, y):
        self.model.train()

    def predict(self, X):
        return self.model.test()

    def score(self, X, y):
        # Implement a scoring method here based on your needs
        pass

def read_data(input_path):
    """Read data for forecasting memory usage.

    Args:
        input_path (str): Path to the dataset.

    Returns:
        X (np.ndarray): Features.
        y (np.ndarray): Ground truth (memory usage percentage).
    """
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date']).map(datetime.toordinal)

    # Assuming you want to predict 'Memory_Usage_Percentage'
    target_column = 'Memory_Usage_Percentage'

    # Selecting features: all columns except the target
    X = df.loc[:, df.columns != target_column].values

    # Target variable
    y = np.array(df[target_column])

    return X, y

dataroot = 'example_time_series_data_corrected.csv'

# Read dataset
print("==> Load dataset ...")
X, y = read_data(dataroot)

# Define the parameter grid
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'nhidden_encoder': [64, 128, 256],
    'nhidden_decoder': [64, 128, 256],
    'batchsize': [32, 64, 128],
    'epochs': [10, 50, 100]
}
batchsize = 128
nhidden_encoder = 128
nhidden_decoder = 128
ntimestep = 10
lr = 0.001
epochs = 50

# Initialize the model
model = RNN(
    X,
    y,
    ntimestep,
    nhidden_encoder,
    nhidden_decoder,
    batchsize,
    lr,
    epochs
)

# Initialize the wrapper
wrapper = PyTorchGrid(model, param_grid)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=wrapper, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV to your data
grid_search.fit(X, y)

# Print the best parameters
print(grid_search.best_params_)