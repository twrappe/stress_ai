import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib  # for saving model

def train_model(features_csv="data/processed/features.csv", model_path="models/stress_classifier.pkl"):
    # Load features
    df = pd.read_csv(features_csv) 
    X = df.drop("label", axis=1)
    y = df["label"]
    
    # Split dataset (stratify to maintain label balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict & evaluate
    y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    
    # Save model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
