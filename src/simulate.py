import pandas as pd
import numpy as np
import os

def generate_gsr_sequence(mean, std, noise=0.01, length=20):
    base = np.random.normal(mean, std, length)
    noise = np.random.normal(0, noise, length)
    return base + noise

def generate_simulated_gsr_csv(path="data/raw/simulated_stress_data.csv"):
    rows = []
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for label, mean, std, start_id in [('calm', 0.25, 0.02, 1), ('stressed', 0.42, 0.03, 101)]:
        for session in range(50):
            session_id = start_id + session
            gsr = generate_gsr_sequence(mean, std)
            for i, value in enumerate(gsr):
                rows.append({
                    'timestamp': round(i * 0.1, 2),
                    'gsr': round(value, 4),
                    'session_id': session_id,
                    'label': label
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✅ Simulated raw data saved to: {path}")

if __name__ == "__main__":
    generate_simulated_gsr_csv()
