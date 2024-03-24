import pandas as pd
import numpy as np

def synthetic_data():
    # Generate a date range for 13 months daily data
    date_range = pd.date_range(start='2020-01-01', end='2021-02-01', freq='D')

    data = pd.DataFrame({
        'timestamp': date_range,
        'cpu_utilization': np.random.randint(10, 100, size=len(date_range)),
        'mem_utilization': np.random.randint(1, 10, size=len(date_range)),
        'cpu_cores': [4]*len(date_range),
        'mem_allocated': [16]*len(date_range),
        'outage_planned': np.random.randint(0, 2, size=len(date_range))  # Randomly generated, replace with your actual data
    })

    # Add a 'holiday' column: 1 if the day is Saturday or Sunday, 0 otherwise
    data['holiday'] = data['timestamp'].apply(lambda x: 1 if x.weekday() >= 5 else 0)

    # Add a 'maintenance' column: 1 if the day is the first day of a quarter, 0 otherwise
    data['maintenance'] = data['timestamp'].apply(lambda x: 1 if (x.is_quarter_start) else 0)

    config = {
        'target': 'cpu_utilization',
        'static_categorical_vars': [],
        'static_real_vars': ['cpu_cores', 'mem_allocated'],
        'known_categorical_vars': ['holiday', 'maintenance'],
        'known_real_vars': ['mem_utilization'],
        'time_var': 'timestamp'
    }

    return data, config

    #processed_data = build_tft_data(data, config)