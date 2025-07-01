import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from src.features import extract_features
import joblib

# Load trained model
model = joblib.load("models/stress_classifier.pkl")

# Constants
WINDOW_SIZE = 20
STREAM_INTERVAL = 0.1  # seconds
MAX_HISTORY = 200

# Initialize buffers
gsr_buffer = deque(maxlen=WINDOW_SIZE)
time_buffer = deque(maxlen=WINDOW_SIZE)
history_gsr = deque(maxlen=MAX_HISTORY)
history_time = deque(maxlen=MAX_HISTORY)
history_pred = deque(maxlen=MAX_HISTORY)

# Simulate GSR input (replace this with real sensor read later)
def simulate_gsr_sample(i):
    # Calm for first 10s, then stressed
    mean, std = (0.25, 0.02) if i < 100 else (0.42, 0.03)
    return np.random.normal(mean, std)

# Set up plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=2)
state_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update_plot():
    line.set_data(history_time, history_gsr)
    ax.set_xlim(max(0, history_time[0]), history_time[-1] + 0.1)
    ax.set_ylim(0.2, 0.5)
    state_text.set_text(f"Predicted State: {history_pred[-1]}")
    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    print("Starting real-time plot...")
    for i in range(200):  # ~20 seconds of data
        gsr = simulate_gsr_sample(i)
        t = round(i * STREAM_INTERVAL, 2)

        gsr_buffer.append(gsr)
        time_buffer.append(t)
        history_gsr.append(gsr)
        history_time.append(t)

        if len(gsr_buffer) == WINDOW_SIZE:
            # Create dummy DataFrame window for feature extraction
            df_window = pd.DataFrame({
                "timestamp": list(time_buffer),
                "gsr": list(gsr_buffer),
                "session_id": [0] * WINDOW_SIZE,
                "label": ["unknown"] * WINDOW_SIZE
            })
            features_df = extract_features(df_window)
            X = features_df.drop("label", axis=1)
            pred = model.predict(X)[0]
            history_pred.append(pred)
        else:
            history_pred.append("...")

        update_plot()
        time.sleep(STREAM_INTERVAL)

    print("Finished.")

if __name__ == "__main__":
    main()
