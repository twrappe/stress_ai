import time
import pandas as pd
import numpy as np
import joblib
from features import extract_features  # reuse feature extraction logic for windows

# Load trained model
model = joblib.load("models/stress_classifier.pkl")

# Simulate incoming GSR sensor data (replace this with real data later)
def simulate_gsr_stream(length=200, label="calm"):
    mean, std = (0.25, 0.02) if label == "calm" else (0.42, 0.03)
    base = np.random.normal(mean, std, length)
    noise = np.random.normal(0, 0.01, length)
    return base + noise

def main():
    WINDOW_SIZE = 20
    STREAM_INTERVAL = 0.1  # seconds per sample
    
    gsr_buffer = []
    timestamps = []
    
    # Use "calm" then "stressed" segments
    stream_data = np.concatenate([
        simulate_gsr_stream(100, "calm"),
        simulate_gsr_stream(100, "stressed")
    ])
    
    for i, gsr_value in enumerate(stream_data):
        gsr_buffer.append(gsr_value)
        timestamps.append(round(i * STREAM_INTERVAL, 2))
        
        if len(gsr_buffer) == WINDOW_SIZE:
            # Prepare dataframe for feature extraction
            df_window = pd.DataFrame({
                "timestamp": timestamps[-WINDOW_SIZE:],
                "gsr": gsr_buffer[-WINDOW_SIZE:],
                "session_id": [0]*WINDOW_SIZE,
                "label": ["unknown"]*WINDOW_SIZE
            })
            
            # Extract features (returns one row dataframe)
            features_df = extract_features(df_window, window_size=WINDOW_SIZE)
            X = features_df.drop("label", axis=1)
            
            # Predict stress
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            print(f"Time {timestamps[-1]:.2f}s - Predicted state: {pred} - (Confidence: {max(proba)*100:.1f}%)")

            # Slide window by one sample (optional)
            gsr_buffer.pop(0)
            timestamps.pop(0)
        
        time.sleep(STREAM_INTERVAL)

if __name__ == "__main__":
    main()
