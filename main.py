from tft_module.tft_optuna import optimize_hyperparameters
from tft_module.input_data import synthetic_data
from tft_module.tft_predict import make_predictions
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # Get the data and configuration
    data, config = synthetic_data()

    # Optimize the hyperparameters and get the best model
    study = optimize_hyperparameters(data, config)
    best_model = study.best_trial.user_attrs["model"]

    # Make predictions using the best model
    predictions = make_predictions(best_model, data)
    print(predictions)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
