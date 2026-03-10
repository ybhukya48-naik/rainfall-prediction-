from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from advanced_data_utils import load_data, preprocess_data
import os

def train_advanced_model():
    """Train an advanced stacking hybrid model (XGBoost + LightGBM + CatBoost)."""
    # Ensure data exists
    data_path = 'data/advanced_weather_data.csv'
    if not os.path.exists(data_path):
        from advanced_data_utils import generate_advanced_dummy_data
        generate_advanced_dummy_data(data_path)
    
    # Load and preprocess
    df = load_data(data_path)
    X, y, features = preprocess_data(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models
    estimators = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('cat', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, verbose=0, random_state=42))
    ]
    
    # Define stacking classifier with logistic regression as final estimator
    hybrid_model = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    print("Training Hybrid Stacking Model (XGBoost, LightGBM, CatBoost)...")
    hybrid_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = hybrid_model.predict(X_test)
    print("\nAdvanced Hybrid Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(hybrid_model, 'models/advanced_hybrid_model.pkl')
    print("Hybrid model saved to models/advanced_hybrid_model.pkl")

if __name__ == "__main__":
    train_advanced_model()
