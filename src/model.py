from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_utils import load_data, preprocess_data
import os

def train_model():
    """Train the rainfall prediction model."""
    # Ensure data exists
    data_path = 'data/weather_data.csv'
    if not os.path.exists(data_path):
        from data_utils import generate_dummy_data
        generate_dummy_data(data_path)
    
    # Load and preprocess
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/rainfall_model.pkl')
    print("Model saved to models/rainfall_model.pkl")

if __name__ == "__main__":
    train_model()
