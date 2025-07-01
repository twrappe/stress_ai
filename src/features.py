import numpy as np
import pandas as pd

def extract_features(df, window_size=20):
    """
    Extract features from raw GSR data grouped by session and windowed by window_size.
    Assumes df has columns: ['timestamp', 'gsr', 'session_id', 'label'].
    
    Returns a DataFrame with features + label per window.
    """
    feature_rows = []

    for session_id, session_data in df.groupby('session_id'):
        gsr_values = session_data['gsr'].values
        labels = session_data['label'].values
        
        # Slide over windows
        for start in range(0, len(gsr_values) - window_size + 1, window_size):
            window = gsr_values[start:start + window_size]
            label = labels[start + window_size // 2]  # label at center of window
            
            features = {
                'mean': np.mean(window),
                'std': np.std(window),
                'max': np.max(window),
                'min': np.min(window),
                'range': np.ptp(window),  # max - min
                'median': np.median(window),
                'label': label
            }
            feature_rows.append(features)
    
    return pd.DataFrame(feature_rows)
