import gc
import base64
import joblib  
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm


st.set_page_config(page_title="Telecom Customer Churn Analytics Platform", layout="wide")
st.title('📊 Telecom Customer Churn Analytics Platform')
st.markdown("#### *Developed by Sachin Kumar | Data Science Dual Degree Program*")

st.markdown("""
Customer churn—the rate at which subscribers discontinue their services—is a critical operational metric for 
modern telecommunication enterprises. 

This enterprise-grade intelligence dashboard utilizes optimized gradient boosting ensembles (**XGBoost, CatBoost, and LightGBM**) 
to predict real-time customer attrition probabilities based on subscriber demographics, account information, and usage metrics.

### 📋 Operational Instructions
1. **Model Selection:** Use the dropdown selector in the sidebar to switch between different optimized gradient boosting engines.
2. **Model Performance:** Click **`Performance on Test Dataset`** in the sidebar to generate real-time ROC-AUC curve visualizations.
3. **Random Audit:** Click **`Prediction on Random Instance from Test Data`** to pick a baseline customer profile and audit predictions.
4. **Custom Prediction:** Adjust the sliders and dropdown menus in the **User Input** sidebar and click **`Predict`** to calculate an custom subscriber's risk.

---

### 🌐 Project Assets & Replication Links
* **Production Codebase:** [![](https://img.shields.io/badge/Production%20Repository-GitHub-100000?logo=github&logoColor=white)](https://github.com/tetarwalsachinkumar/customer-churn-prediction)
* **Pipeline Step 1:** [Notebook I: Data Engineering & Generation Pipeline](https://www.kaggle.com/sachin211104/customer-churn-prediction-i-data-generation)
* **Pipeline Step 2:** [Notebook II: Stacking Ensemble & Optuna Hyperparameter Tuning](https://www.kaggle.com/sachin211104/customer-churn-prediction-ii-model)
""")

# 📁 Safety Patch: Standardized Unix paths to prevent directory execution errors on Streamlit Cloud
df_churn = pd.read_csv("dataset/Telco-Customer-Churn-dataset-cleaned.csv")
df_train = pd.read_csv('dataset/Telco-Customer-Churn-dataset-Train.csv', index_col=0)
df_test = pd.read_csv('dataset/Telco-Customer-Churn-dataset-Test.csv', index_col=0)

st.header('🔍 Churn Dataset Overview')
st.write(f'**Active Ledger Metrics:** Evaluated data snapshot contains `{df_churn.shape[0]}` historical rows and `{df_churn.shape[1]}` analytical columns.')
st.dataframe(df_churn)


@st.cache_data
def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv" style="text-decoration:none;"><button style="background-color:#4CAF50; border:none; color:white; padding:8px 16px; text-align:center; text-size:14px; border-radius:4px; cursor:pointer;">📥 Export Dataset as CSV</button></a>'
    return href

st.markdown(download_dataset(df_churn), unsafe_allow_html=True)
st.markdown("---")

st.markdown("## 🎯 Engine Prediction Result")

st.sidebar.markdown("## ⚙️ Model Control Center")
classifier_name = st.sidebar.selectbox(
    'Select Target Predictive Classifier',
    ('XGBoost', 'CatBoost', 'LightGBM')
)


def get_classifier(clf_name):
    if clf_name == 'XGBoost':
        clf = xgb.XGBClassifier()  # init model
        clf.load_model("models/model_xgb.json")
    elif clf_name == 'CatBoost':
        clf = CatBoostClassifier()  # parameters not required.
        clf.load_model('models/model_catboost')
    else:
        # 🛠️ Fix: Successfully loads your optimized .pkl binary instead of failing on a missing .txt layout
        clf = joblib.load("models/model_lgbm.pkl")
    return clf


clf = get_classifier(classifier_name)


def get_transformed_data(test_data=None):
    X = df_train.drop("Churn", axis=1)

    if test_data is None:
        test_data = df_test.copy()
    # test dataset
    y_test = test_data['Churn'].values
    X_test = test_data.drop("Churn", axis=1)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = list(set(X.columns) - set(X._get_numeric_data().columns))

    ordinal_encoder = OrdinalEncoder()
    X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

    transformer = RobustScaler()
    X[num_cols] = transformer.fit_transform(X[num_cols])
    X_test[num_cols] = transformer.transform(X_test[num_cols])

    del X
    gc.collect()
    return X_test, y_test


def make_prediction(X_test):
    try:
        # standard call for scikit-learn interface models (XGBoost, CatBoost, or LGBMClassifier wrapper)
        test_pred = clf.predict_proba(X_test)[:, 1]  # probability of getting 1
    except AttributeError:
        # backup fallback engine mapping for raw native LightGBM boosters
        test_pred = clf.predict(X_test)
    return test_pred


if st.sidebar.button('Performance on Test Dataset'):
    X_test, y_test = get_transformed_data()
    test_pred = make_prediction(X_test)
    
    st.markdown(f"### 📈 Diagnostics: {classifier_name} Engine")
    st.info(f"**Calculated ROC-AUC Target Validation Accuracy Score:** `{roc_auc_score(y_test, test_pred):.5f}`")

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f'Receiver Operating Characteristic ({classifier_name})', fontsize=12)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    st.pyplot(fig)

if st.sidebar.button('Prediction on Random Instance from Test Data'):
    random_test_instance = df_test.sample(n=1)
    X_test, y_test = get_transformed_data(random_test_instance)
    test_pred = make_prediction(X_test)
    
    st.markdown("### 🎲 Random Audit Results")
    status_label = '🔴 HIGH RISK: CHURNED' if test_pred[0] > 0.5 else '🟢 STABLE: ACTIVE SUBSCRIBER'
    st.subheader(f"Status Evaluation: {status_label}")
    st.metric(label="Calculated Attrition Probability Score", value=f"{test_pred[0]*100:.2f}%")
    st.write("**Audited Customer Record Attributes:**")
    st.dataframe(random_test_instance)


st.sidebar.markdown('## 🖋️ Custom Subscriber Matrix')


def binning_feature(feature, value):
    bins = np.linspace(min(df_churn[feature]), max(df_churn[feature]), 4)
    if bins[0] <= value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'


def user_input_features():
    gender = st.sidebar.selectbox('Gender Demographics', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen Status', ('No', 'Yes'))
    partner = st.sidebar.selectbox('Has Registered Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Has Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service Allocation', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines Provisioning', ('No', 'Yes', 'No phone service'))
    internet_service_type = st.sidebar.selectbox('Internet Service Architecture', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Value Added Product: Online Security', ('No', 'Yes', 'No internet service'))
    online_backup = st.sidebar.selectbox('Value Added Product: Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Value Added Product: Device Protection', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Value Added Product: Premium Tech Support', ('No', 'Yes', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Entertainment Feed: Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Entertainment Feed: Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Account Billing Structure Contract Type', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Invoicing Choice', ('Yes', 'No'))

    payment_method = st.sidebar.selectbox('Preferred Settlement Channel', (
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    # tenure filter
    unique_tenure_values = df_churn.tenure.unique()
    min_value, max_value = min(unique_tenure_values), max(unique_tenure_values)
    tenure = st.sidebar.slider("Account Tenure Lifecycle (Months)", int(min_value), int(max_value), int(min_value), 1)

    # MonthlyCharges filter
    unique_monthly_charges_values = df_churn.MonthlyCharges.unique()
    min_value, max_value = min(unique_monthly_charges_values), max(unique_monthly_charges_values)
    monthly_charges = st.sidebar.slider("Assigned Monthly Flat Charges ($)", float(min_value), float(max_value), float(min_value))

    min_value_total = monthly_charges * tenure
    max_value_total = (monthly_charges * tenure) + 100

    st.sidebar.caption("**Financial Estimate Formula:** `TotalCharges = (MonthlyCharges * Tenure) + Overhead`")
    total_charges = st.sidebar.slider("Calculated Accumulative Total Charges ($)", float(min_value_total), float(max_value_total))

    data = {'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen.lower() == 'yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service_type],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure-binned': binning_feature('tenure', tenure),
            'MonthlyCharges-binned': binning_feature('MonthlyCharges', monthly_charges),
            'TotalCharges-binned': binning_feature('TotalCharges', total_charges)
            }

    features = pd.DataFrame(data)
    return features


input_df = user_input_features()

num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = input_df.select_dtypes(include=['object']).columns

# Fit real-time transforms via robust pipelines
X = df_train.drop("Churn", axis=1)
user_df = input_df.copy()
ordinal_encoder = OrdinalEncoder()
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
user_df[cat_cols] = ordinal_encoder.transform(user_df[cat_cols])

transformer = RobustScaler()
X[num_cols] = transformer.fit_transform(X[num_cols])
user_df[num_cols] = transformer.transform(user_df[num_cols])

if st.sidebar.button('Predict Attrition Probability'):
    test_pred = make_prediction(user_df)
    
    st.markdown("### 📋 Interactive User Profile Analysis")
    custom_status = '🔴 HIGH ATTRITION RISK (CHURN)' if test_pred[0] > 0.5 else '🟢 RISK MINIMAL (RETAINED)'
    st.subheader(f"Engine Decision: {custom_status}")
    st.metric(label="Calculated Profile Churn Score", value=f"{test_pred[0]*100:.2f}%")
    st.write("**Evaluated Inputs Configuration Matrix:**")
    st.dataframe(input_df)