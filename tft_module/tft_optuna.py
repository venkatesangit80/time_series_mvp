import optuna
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from tft_module.data_builder import build_tft_data

def create_and_train_tft_model(trial, dataset):
    """
    This function creates and trains a Temporal Fusion Transformer model with hyperparameters specified by an Optuna trial.

    Parameters:
    trial (optuna.trial.FrozenTrial): The Optuna trial that specifies the hyperparameters.
    dataset (TimeSeriesDataSet): The dataset to train the model on.

    Returns:
    float: The validation loss of the trained model.
    """

    # Define the hyperparameters
    hyperparameters = {
        "hidden_size": trial.suggest_int("hidden_size", 16, 512),
        "attention_head_size": trial.suggest_int("attention_head_size", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 8, 128),
        "output_size": trial.suggest_int("output_size", 7, 28),  # forecast horizon
        "loss": SMAPE(),
        "log_interval": 10,  # log example every x steps
        "reduce_on_plateau_patience": 4,  # reduce learning rate if no improvement in validation loss after x epochs
    }

    # Create the model
    model = TemporalFusionTransformer.from_dataset(dataset, **hyperparameters)

    # Train the model
    model.fit()

    # Return the validation loss
    return model.best_validation_loss

def objective(trial, dataset):
    """
    This function defines the objective for the Optuna study.

    Parameters:
    trial (optuna.trial.FrozenTrial): The Optuna trial that specifies the hyperparameters.
    dataset (TimeSeriesDataSet): The dataset to train the model on.

    Returns:
    float: The validation loss of the trained model.
    """

    return create_and_train_tft_model(trial, dataset)

def optimize_hyperparameters(data, config):
    """
    This function optimizes the hyperparameters of a Temporal Fusion Transformer model using Optuna.

    Parameters:
    data (pd.DataFrame): The raw data. This should be a DataFrame with columns for different features and a timestamp.
    config (dict): A dictionary containing configuration options for processing the data.

    Returns:
    optuna.study.Study: The Optuna study that contains the results of the hyperparameter optimization.
    """

    # Process the data
    processed_data = build_tft_data(data, config)

    # Create a TimeSeriesDataSet from the processed data
    dataset = TimeSeriesDataSet(processed_data, ...)

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize the hyperparameters
    study.optimize(lambda trial: objective(trial, dataset), n_trials=100)

    return study