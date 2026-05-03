Live App: [Check it out here!](https://mystudentscorepredictionmodel.streamlit.app/)

# Student Score Prediction Model

Predicts student exam scores using XGBoost based on Study Methodology (Offline/Online/Hybrid), Age, and Study Hours.

##  How to Use
1. Upload a CSV file or enter data manually.
2. Ensure your CSV has three columns: **Methodology**, **Age**, and **Study Hours**.
3. Click "Predict" to see the results.

##  Tech Stack
* **Model:** XGBoost Regressor
* **Preprocessing:** Scikit-Learn Pipeline (OneHotEncoder & SimpleImputer)
* **Frontend:** Streamlit
* **Deployment:** Streamlit Community Cloud

##  Requirements
The methodology column must only contain: `Offline`, `Online`, or `Hybrid`.
