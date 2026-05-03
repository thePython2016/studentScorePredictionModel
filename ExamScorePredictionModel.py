import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pickle

# pickle.dump(gModel,open("gModel.pkl","wb"))
# # pickle.dump(combined,open("combined.pkl","wb"))

gModel=pickle.load(open("model.pkl","rb"))
combined=pickle.load(open("transformer.pkl","rb"))
st.title("Student Score Prediction Model")

tab1,tab2=st.tabs(['User Input','Upload File'])


with tab1:
    Methodology=st.text_input("Enter Teaching Methodology  Eg Offline, Online, Hybrid",placeholder="Offline, Online, Hybrid")
    Age=st.number_input("Enter Your Age")
    Hours=st.number_input("Enter Study Hours")
    button=st.button("predict")

    if button:
        allowed_methods = ["Offline", "Online", "Hybrid"]
        if Methodology.title() not in allowed_methods:
            st.error("Methodology must be either Offline, Online, or Hybrid.")
        elif Age<=0:
                st.error("Enter Real Age")
        elif Hours<=0:
                st.error("Enter Real Hours")
        else:
                df=pd.DataFrame({"Methodology":[Methodology],"Age":[Age],"Study Hours":[Hours]})
                transform=combined.transform(df)
                predict=gModel.predict(transform)
                actualScore=np.expm1(predict)
                st.success(f"Predicted Score: {actualScore}")
            
with tab2:
    file = st.file_uploader("upload CSV", type="csv")
    buttonFile = st.button("Predict")
    
    if buttonFile:
        if file is None:
            st.error("Please Upload a File")
        else:
            try:
                # 1. File Processing the file
                fileFrame = pd.read_csv(file)
                
                # 2. Handling outliers 
                for col in fileFrame.select_dtypes(include=["float64", "int64"]).columns:
                    Q1 = fileFrame[col].quantile(0.25)
                    Q3 = fileFrame[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0 or pd.isna(IQR):
                        continue
                    Lower = Q1 - 1.5 * IQR
                    Upper = Q3 + 1.5 * IQR
                    fileFrame[col] = fileFrame[col].clip(Lower, Upper)

                # 3. Transforming  and Predicting 
                fileTransform = combined.transform(fileFrame)
                predictFile = gModel.predict(fileTransform)
                actualScore = np.expm1(predictFile)
                
                # 4. Handling  results
                if len(actualScore) == 1:
                    st.success(f"Predicted Score: {actualScore[0]:.2f}")
                else:
                    fileFrame['Predicted Score'] = actualScore.round(2)
                    st.success(f"Successfully predicted scores for {len(fileFrame)} rows!")
                    st.dataframe(fileFrame)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Check if your CSV column names match: Name, Age, Study Hours, Methodology")

    st.info("""
    ### Data Requirements
    To ensure accurate predictions, please make sure your uploaded CSV file contains exactly three columns: 
    **Methodology**, **Age**, and **Study Hours**. 

    **Important:**
    *   The **Methodology** values must be exactly: **Offline**, **Online**, or **Hybrid**.
    *   Column names are case-sensitive. 
    *   Providing unauthorized values or incorrect column names will cause the prediction to fail.
    """)



