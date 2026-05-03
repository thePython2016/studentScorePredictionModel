import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle

# pickle.dump(gModel,open("gModel.pkl","wb"))
# # pickle.dump(combined,open("combined.pkl","wb"))

gModel=pickle.load(open("model.pkl","rb"))
combined=pickle.load(open("transformer.pkl","rb"))
st.title("Score Prediction Model")

tab1,tab2=st.tabs(['User Input','Upload File'])


with tab1:
    name=st.text_input("Enter your name")
    Age=st.number_input("Enter Your Age")
    Hours=st.number_input("Enter Study Hours")
    button=st.button("predict")

    if button:
        if name.isnumeric():
            st.error("Enter real name")
        elif Age<=0:
              st.error("Enter Real Age")
        elif Hours<=0:
               st.error("Enter Real Hours")
        else:
            df=pd.DataFrame({"Name":[name],"Age":[Age],"Study Hours":[Hours]})
            transform=combined.transform(df)
            predict=gModel.predict(transform)
            actualScore=np.expm1(predict)
            st.success(actualScore)
            
with tab2:
    file=st.file_uploader("upload CSV",type="csv")
    buttonFile=st.button("Predict")
    if buttonFile:
        if file is None:
            st.error("Upload File")
        else:
            # dataFile=file.value[0]['content']
            fileFrame=pd.read_csv(file)
            fileTransform=combined.transform(fileFrame)
            predictFile=gModel.predict(fileTransform)
            actualScore=np.expm1(predictFile)
            st.success(actualScore)





