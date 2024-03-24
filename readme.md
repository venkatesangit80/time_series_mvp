# Temporal Fusion Transformer with Optuna

This project uses the Temporal Fusion Transformer (TFT) model for time series forecasting, with hyperparameters optimized using Optuna.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```
## Usage

The main components of the project are:

- `tft_optuna.py`: This module contains the code for creating and training a TFT model with hyperparameters specified by an Optuna trial. It also defines the objective for the Optuna study and a function to optimize the hyperparameters.

- `tft_predict.py`: This module contains a function for making predictions using a trained TFT model.

- `input_data.py`: This module generates synthetic data and defines known categorical variables.

To use the project, first optimize the hyperparameters and train the model:

```python
from tft_module.tft_optuna import optimize_hyperparameters
from tft_module.input_data import synthetic_data

# Get the data and configuration
data, config = synthetic_data()

# Optimize the hyperparameters and get the best model
study = optimize_hyperparameters(data, config)
best_model = study.best_trial.user_attrs["model"]
```

Then, use the trained model to make predictions:

```python
from tft_module.tft_predict import make_predictions

# Make predictions
predictions = make_predictions(best_model, dataset)
```


Replace dataset with the actual dataset you want to make predictions on.