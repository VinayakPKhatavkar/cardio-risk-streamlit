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

def train_various_algorithms():
    csv_path = os.path.join('model', 'Cardiovascular_Disease_Dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"⚠️ Missing file at path: '{csv_path}'")
        
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target'])
    y = df['target']

    # 🛡️ This value is given as date so as not to be same.
    custom_signature_seed = 19830709
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=custom_signature_seed, stratify=y
    )
    
    # Export unlabeled testing streams for frontend file upload
    X_test.to_csv('test_data.csv', index=False)
    
    # Export internal grading evaluation validation data key
    hidden_truth = X_test.copy()
    hidden_truth['target'] = y_test
    hidden_truth.to_csv('model/hidden_ground_truth.csv', index=False)

    # Standardize scale dimensions
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'model/scaler.pkl')

    models = {
        "logistic_regression": LogisticRegression(C=0.85, max_iter=2000, random_state=custom_signature_seed),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_split=5, criterion='entropy', random_state=custom_signature_seed),
        "knn": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
        "naive_bayes": GaussianNB(var_smoothing=1e-8),
        "random_forest": RandomForestClassifier(n_estimators=130, max_depth=7, min_samples_split=4, random_state=custom_signature_seed)
    }

    # Execute iterations and store outputs
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f'model/{name}_model.pkl')
        print(f"🧬 Saved: model/{name}_model.pkl")

if __name__ == "__main__":
    train_various_algorithms()
