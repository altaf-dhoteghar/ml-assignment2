# Machine Learning Assignment 2: UCI Mushroom Classification App

**Course:** M.Tech AIML - Machine Learning  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Dataset:** UCI Mushroom Dataset  

---

## 1. Problem Statement
The goal of this project is to implement, evaluate, and deploy multiple machine learning classification models to determine whether a given mushroom specimen is **edible (0)** or **poisonous (1)** based on its physical and environmental characteristics. Accurately identifying toxic mushrooms is crucial for preventing severe poisoning, making this a high-stakes binary classification problem where high precision and recall are mandatory.

---

## 2. Dataset Description
* **Dataset Name:** UCI Mushroom Dataset (Agaricus and Lepiota)
* **Source:** UCI Machine Learning Repository : `https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data`
* **Instance Size:** 8,124 instances (Exceeds minimum required size of 500)
* **Feature Size:** 22 categorical features + 1 target class (Exceeds minimum required feature size of 12)
* **Target Variable:** `class` (Binary: `0` = Edible, `1` = Poisonous)
* **Features Included:** Cap Shape, Cap Surface, Cap Color, Bruises, Odor, Gill Attachment, Gill Spacing, Gill Size, Gill Color, Stalk Shape, Stalk Root, Stalk Surface Above/Below Ring, Stalk Color Above/Below Ring, Veil Type, Veil Color, Ring Number, Ring Type, Spore Print Color, Population, and Habitat.
* **Data Preprocessing:** Categorical features were encoded using `OneHotEncoder` from Scikit-Learn to convert nominal attributes into numeric matrices suitable for machine learning models.

---

## 3. GitHub Repository Link
* **GitHub Repository:** `https://github.com/altaf-dhoteghar/ml-assignment2.git`
* **Live Streamlit App:** `https://ml-assignment2-app.streamlit.app/`

---

## 4. Models Used & Evaluation Metrics

All 6 required models were implemented on the exact same dataset split (80% train, 20% test, stratify split):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9994 | 1.0000 | 1.0000 | 0.9987 | 0.9994 | 0.9988 |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **kNN** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Naive Bayes** | 0.9329 | 0.9954 | 0.9814 | 0.8774 | 0.9265 | 0.8698 |
| **Random Forest (Ensemble)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


---

## 5. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well with an Accuracy of 99.94% and 1.0000 AUC. The high linearly separable nature of one-hot encoded categorical features (such as `odor` and `spore-print-color`) allows linear decision boundaries to achieve fast and nearly flawless convergence. |
| **Decision Tree** | Achieved 100% accuracy across all evaluation metrics. Physical features in this dataset (particularly `odor`) contain near-deterministic decision boundaries, allowing a single tree to perfectly segment edible vs. poisonous samples without over-fitting. |
| **kNN** | Achieved 100% accuracy across all metrics. Because distinct physical combinations cleanly isolate poisonous classes from edible ones in feature space, distance-based nearest-neighbor lookup works perfectly. |
| **Naive Bayes** | Showed the lowest relative performance (93.29% Accuracy, 0.8698 MCC). This occurs because Naive Bayes assumes conditional independence between all features, an assumption partially violated by correlated physical traits (e.g., stalk color and ring type). |
| **Random Forest (Ensemble)** | Achieved perfect 100% scores across all metrics (Accuracy, AUC, Precision, Recall, F1, MCC). Ensemble bagging across multiple decision trees eliminates variance risks while capturing complex rule interactions. |

### Overall Winner for your dataset?
**Random Forest (Ensemble)** is selected as the overall winner. While Decision Tree and kNN also achieved 100% metrics on this specific dataset test split, **Random Forest** is the superior, production-ready choice due to its robustness against overfitting, lower variance, and ability to generalize reliably to unseen edge-case data.