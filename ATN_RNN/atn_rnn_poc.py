import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from module.RNN import RNN
from datetime import datetime


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

batchsize = 128
nhidden_encoder = 128
nhidden_decoder = 128
ntimestep = 10
lr = 0.001
epochs = 50

# Read dataset
print("==> Load dataset ...")
X, y = read_data(dataroot)

# Initialize model
print("==> Initialize DA-RNN model ...")
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

# Train
print("==> Start training ...")
model.train()

# Prediction
y_pred = model.test()

fig1 = plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.savefig("1.png")
plt.close(fig1)

fig2 = plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.savefig("2.png")
plt.close(fig2)

fig3 = plt.figure()
plt.plot(y_pred, label='Predicted')
plt.plot(model.y[model.train_timesteps:], label="True")
plt.legend(loc='upper left')
plt.savefig("3.png")
plt.close(fig3)
print('Finished Training')