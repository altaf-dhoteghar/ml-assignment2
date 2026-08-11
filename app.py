import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def fetch_n_prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
    ]

    data = pd.read_csv(url, names=columns)
    data.to_csv("live_dataset.csv")
    print("dataset downloaded from UCI")

    # p => poisonous, e => edible
    data['class'] = data['class'].map({'p':1, 'e': 0})

    x = data.drop("class", axis=1)
    y = data["class"]

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    test_data = x_test_raw.copy()
    test_data['class'] = y_test
    test_data.to_csv('test_data.csv', index=False)
    print("test data set created")

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output = False)
    x_train = ohe.fit_transform(x_train_raw)
    x_test = ohe.transform(x_test_raw)

    return x_train, x_test, y_train,y_test, test_data, ohe


x_train, x_test, y_train,y_test, test_data, ohe = fetch_n_prep_data()

@st.cache_resource
def train_all(x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": CategoricalNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_probablity = model.predict_proba(x_test)[:,1]
        else:
            y_probablity = y_predict


        accuracy = accuracy_score(y_test, y_predict)
        auc = roc_auc_score(y_test, y_probablity)
        prec = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        mathew = matthews_corrcoef(y_test, y_predict)

        results[name] = {
            "model": model, 
            "y_predict": y_predict, 
            "metrics": {
                "accuracy": accuracy,
                "AUC": auc, 
                "Precision": prec,
                "Recall": recall,
                "F1": f1,
                "Mathews CorrCoef": mathew
            }
        }
    return results
results = train_all(x_train, x_test, y_train, y_test)

# ---------------------------------------------------------
# Streamlit Web Interface Controls
# ---------------------------------------------------------
st.set_page_config(page_title="UCI Mushroom Classification App", layout="wide")
st.title("UCI Mushroom Dataset - ML Classification Dashboard")

# side bar contain file upload
st.sidebar.header("Options: Select Model or upload test data")
st.sidebar.markdown("---")
upload_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
if upload_file is not None:
    st.sidebar.success("Custom test dataset loaded!")
    upload_data = pd.read_csv(upload_file)
    st.write("Uploaded Test Data")
    st.dataframe(upload_data.head())
    data_x = upload_data.drop("class",axis=1)
    ohe.transform(data_x)
    

st.sidebar.markdown("---")
with open("test_data.csv","rb") as file:
    st.sidebar.download_button(
        label="Downlaod Sample Test Data",
        data=file,
        file_name="mashroom_sample_data.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
# side bar also contain dropdown for modules
model_name = st.sidebar.selectbox("Select ML Model:", list(results.keys()))
st.sidebar.markdown("---")



st.markdown("---")
st.markdown("Implemented as part of Machine Learning Assignment 2.")


st.header(f"Model Results: {model_name}")
model_results = results[model_name]
metrics = model_results["metrics"]
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
col2.metric("AUC", f"{metrics['AUC']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")
col4.metric("Recall", f"{metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{metrics['F1']:.4f}")
col6.metric("Mathews CorrCoef Score", f"{metrics['Mathews CorrCoef']:.4f}")
st.markdown("---")


coll, colr = st.columns(2)

with coll:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, model_results["y_predict"])
    fig, ax = plt.subplots(figsize = (4,3))
    sns.heatmap (cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                 xticklabels=["Edible", "Poisonous"],
                 yticklabels=["Edible", "Poisonous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
with colr:
    st.subheader("Classification Report")
    report = classification_report(y_test, model_results["y_predict"], target_names=["Edible", "Poisonous"])
    st.code(report)