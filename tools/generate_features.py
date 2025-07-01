import pandas as pd
from src.features import extract_features
import os

RAW_DATA_PATH = "data/raw/gsr_data.csv"
PROCESSED_DATA_PATH = "data/processed/features.csv"

def main():
    # Load raw GSR time-series data
    df = pd.read_csv(RAW_DATA_PATH)

    # Extract features from windows
    features_df = extract_features(df, window_size=20)

    # Drop any rows with NaNs (optional safeguard)
    features_df = features_df.dropna()

    # Ensure target directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Save to CSV
    features_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Features saved to: {PROCESSED_DATA_PATH}")
    print(features_df.head())

if __name__ == "__main__":
    main()
