import pandas as pd
import numpy as np
from pytorch_forecasting.metrics import QuantileLoss

date_rng = pd.date_range(start='1/1/2020', end='2/1/2021', freq='D')
cpu_utilization = np.random.randint(0, 100, size=(len(date_rng),))

df = pd.DataFrame(date_rng, columns=['date'])
df['cpu_utilization'] = cpu_utilization
df['time_idx'] = df.index
df['month'] = df.date.dt.month.astype(str).astype("category")  # categories have be strings


from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

max_encoder_length = 30
max_prediction_length = 10

training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="cpu_utilization",
    group_ids=["month"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["month"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["cpu_utilization"],
    target_normalizer=GroupNormalizer(groups=["month"]),
    allow_missing_timesteps=True,  # allow missing time steps
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True, allow_missing_timesteps=True)


batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=8)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner


pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="cpu",
    gradient_clip_val=0.1,
)

# Define your Temporal Fusion Transformer model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,  # Adjust this based on your task
    loss=QuantileLoss(),
    log_interval=-1,
    reduce_on_plateau_patience=4,
)

# Setup PyTorch Lightning trainer
# Here we don't specify any GPU settings, so it defaults to CPU
trainer = pl.Trainer(
    max_epochs=10,  # You might want to start with a lower number of epochs
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(dirpath="./model_checkpoints/", save_top_k=1, monitor="val_loss"),
    ],
)


# Train the model
trainer.fit(
    model=tft,
    train_dataloaders=training.to_dataloader(train=True, batch_size=16, num_workers=0),
    val_dataloaders=validation.to_dataloader(train=False, batch_size=16, num_workers=0),
)

# Optionally, load the best model later
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

val_prediction_results = best_model.predict(validation)
print("Completed")