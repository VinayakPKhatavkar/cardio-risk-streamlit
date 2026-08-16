# cardio-risk-streamlit
## a. Problem statement
Cardiovascular disease tracking requires highly resilient feature evaluations to support clinical screening workflows.

The primary objective is to implement, balance, evaluate, and deploy five distinct machine learning classification models on a clinical diagnostic matrix. This project checks model consistency, performance metrics, and resilience against unseen test data streams via an interactive web interface.

We will be implementing below models:
1. Logistic Regression 
2. Decision Tree Classifier 
3. K-Nearest Neighbor Classifier 
4. Naive Bayes Classifier - Gaussian or Multinomial 
5. Ensemble Model - Random Forest

## b. Dataset Description
The model analytics framework leverages the **Kaggle Heart Disease Dataset** (Source file imported as `Cardiovascular_Disease_Dataset.csv`). It encompasses **1,025 observation points** featuring **14 specific diagnostic attributes**:

*   `age`: Chronological age classification metric (Years).
*   `sex`: Categorical anatomical variable profile tracker (1 = Male; 0 = Female).
*   `cp`: Chest pain expression complexity tier score markers (Range values: 0 to 3).
*   `trestbps`: Resting state cardiovascular system blood pressure tracking indexes (mm Hg).
*   `chol`: Circulating blood lipid profiling element values (mg/dl).
*   `fbs`: Fasting state blood sugar indicator evaluation thresholds (> 120 mg/dl mapping).
*   `restecg`: Resting electrocardiographic electrical measurement trace variations.
*   `thalach`: Peak heart rates achieved across standardized cardiac stress runs.
*   `exang`: Exercise-induced physical discomfort / angina flag metrics.
*   `oldpeak`: Ischemic changes / ST-segment depression profiles relative to base states.
*   `slope`: Morphological slope characteristics of the ST segment vector.
*   `ca`: Fluoroscopy colored principal vessel classification metrics count (0-4 range).
*   `thal`: Thalassemia defect grouping vectors.
*   `target`: Clinical diagnostic classification indicator (0 = Lower risk, 1 = Presence of heart disease risk).

## c. GitHub Repository Link
https://github.com/VinayakPKhatavkar/cardio-risk-streamlit

## d. Machine Learning Classification Evaluation Scores Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| :----------------------- | :------- | :----- | :-------- | :----- | :----- | :----- |
| Logistic Regression      | 0.9600   | 0.9943 | 0.9576    | 0.9741 | 0.9658 | 0.9178 |
| Decision Tree            | 0.9600   | 0.9780 | 0.9737    | 0.9569 | 0.9652 | 0.9184 |
| kNN                      | 0.9650   | 0.9970 | 0.9823    | 0.9569 | 0.9694 | 0.9289 |
| Naive Bayes              | 0.9500   | 0.9888 | 0.9569    | 0.9569 | 0.9569 | 0.8974 |
| Random Forest (Ensemble) | 0.9900   | 0.9995 | 0.9831    | 1.0000 | 0.9915 | 0.9796 |


## e. Observations on Model Performance
The table below details structural observations regarding how each algorithm processed the 14 clinical features from the Cardiovascular Disease Dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Acts as a stable baseline classifier. Since many clinical features (like resting blood pressure and age) exhibit linear trends with the target variable, it achieves high generalizability with a balanced Precision and Recall, avoiding high-variance overfitting. |
| **Decision Tree** | Captures non-linear clinical interactions well (e.g., combining age, sex, and chest pain type rules). However, due to its deep recursive splitting behavior, it shows slight overfitting trends unless structural constraints like `max_depth` are tightly enforced. |
| **kNN** | Highly effective because instances with similar health profiles cluster closely together in multi-dimensional space. Because features were properly transformed via `StandardScaler`, it calculates distance metrics accurately, yielding strong sensitivity (Recall). |
| **Naive Bayes** | Performs predictions quickly by treating all clinical bio-markers as entirely independent. While this "independence assumption" slightly penalizes its final accuracy (since metrics like blood pressure and age correlate), it remains incredibly resilient to noise. |
| **Random Forest (Ensemble)** | Achieves peak optimization performance across the dataset. By combining 130 unique decision tree paths and randomizing sub-features, it reduces structural variance, handles column multi-collinearity smoothly, and yields the highest ROC AUC score. |
| **Overall Winner** | **Random Forest (Ensemble)**. It handles the complex mix of continuous metrics (cholesterol, max heart rate) and categorical variables (sex, chest pain type) better than any single model, delivering the optimal balance between high precision and low false-negative rates. |
