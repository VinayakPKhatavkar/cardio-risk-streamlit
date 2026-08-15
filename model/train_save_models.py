import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def train_kaggle_dataset():
    # 1. Access the specific downloaded file name
    csv_path = os.path.join('model', 'Cardiovascular_Disease_Dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"⚠️ Dataset file missing at: '{csv_path}'. Please check your file spelling and location!")
        
    df = pd.read_csv(csv_path)
    
    # 2. Extract feature-target sets (the class outcome field is named 'target')
    X = df.drop(columns=['target'])
    y = df['target']

    # Stratified Train-Test Split (80% Train, 20% Evaluation) using a custom seed (e.g., 55) to customize metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=55, stratify=y
    )
    
    # Save the unlabeled test dataset file for the Streamlit file uploader field
    X_test.to_csv('test_data.csv', index=False)
    
    # Generate hidden evaluation grounds truth validation target frame
    hidden_truth = X_test.copy()
    hidden_truth['target'] = y_test
    hidden_truth.to_csv('model/hidden_ground_truth.csv', index=False)

    # 3. Fit data scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'model/scaler.pkl')

    # 4. Define the 5 specific assignment-required algorithms
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=55),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=55),
        "knn": KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=6, random_state=55)
    }

    # 5. Fit models and serialize them into the model directory
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f'model/{name}_model.pkl')
        print(f"Successfully configured and saved artifact: model/{name}_model.pkl")

if __name__ == "__main__":
    train_kaggle_dataset()
