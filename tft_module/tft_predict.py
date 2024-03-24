from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

def make_predictions(model: TemporalFusionTransformer, dataset: TimeSeriesDataSet):
    """
    This function makes predictions using a trained Temporal Fusion Transformer model.

    Parameters:
    model (TemporalFusionTransformer): The trained model.
    dataset (TimeSeriesDataSet): The dataset to make predictions on.

    Returns:
    pd.DataFrame: The predictions.
    """

    # Make predictions
    predictions = model.predict(dataset)

    return predictions