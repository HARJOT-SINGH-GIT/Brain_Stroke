import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df=pd.read_csv("full_data.csv")

from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()

df["gender"]=l1.fit_transform(df["gender"])
df["ever_married"]=l1.fit_transform(df["ever_married"])
df["work_type"]=l1.fit_transform(df["work_type"])
df["Residence_type"]=l1.fit_transform(df["Residence_type"])
df["smoking_status"]=l1.fit_transform(df["smoking_status"])

x=df.drop(["stroke"],axis=1).values
y=df["stroke"]


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sc = StandardScaler()
X = sc.fit_transform(x)
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.35)

import streamlit as st

st.write("""# Brain Stroke Predictor 
         This Web App predict the Brain Stroke based on user input parameters""")

st.sidebar.header("User Input Parameters")
def user_input_features():
    sex=st.sidebar.selectbox("Gender",["Male","Female"],index=1)
    if sex=="Male":
        gender=1
    else:
        gender=0
    age=st.sidebar.number_input("age",min_value=0.0,max_value=120.0,step=1.0)
    hy=st.sidebar.selectbox("Hypertension",["Yes","No"],index=1)
    if hy=="Yes":
        hypertension=1
    else:
        hypertension=0
    hd=st.sidebar.selectbox("Heart Disease",["Yes","No"],index=1)
    if hd=="Yes":
        heart_disease=1
    else:
        heart_disease=0
    avg_glucose_level=st.sidebar.number_input('avg_glucose_level',min_value=45.0,max_value=270.0)
    bmi=st.sidebar.number_input("bmi",min_value=10.0,max_value=80.0)
    em=st.sidebar.selectbox("Ever Married",["Yes","No"],index=1)
    if em=="Yes":
        ever_married=1
    else:
        ever_married=0
    ss=st.sidebar.selectbox("smoking ststus",["formely smoked","never smoked","smoked","unknown"],index=1)
    if ss=="formely smoked":
        smoking_ststus=1
    elif ss=="never smoked":
        smoking_ststus=2
    elif ss=="smoked":
        smoking_ststus=3
    else:
        smoking_ststus=4
    wt=st.sidebar.selectbox("Work Type",["Private","Self-employed","Govt_job","children"])
    if wt=="Private":
        work_type=1
    elif wt=="Self-employed":
        work_type=2
    elif wt=="children":
        work_type=3
    else:
        work_type=4
    rt=st.sidebar.selectbox("Residence Type",["Urban","Rural"],index=1)
    if rt=="Urban":
        Residence_type=1
    else:
        Residence_type=0
    data={'Residence_type':Residence_type,
          'gender':gender,
          'work_type':work_type,
          'age':age,
          'hypertension':hypertension,
          'heart_disease':heart_disease,
          'ever_married':ever_married,
          'avg_glucose_level':avg_glucose_level,
          'bmi':bmi,
          'smoking_ststus':smoking_ststus}
    features=pd.DataFrame(data,index=[0])
    return features
uif=user_input_features()
st.subheader('User Input Features')
st.write(uif)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
prediction_dt=dt.predict(uif)
if prediction_dt>=0.5:
    st.write("You hav high chance of getting Brain Stroke")
else:
    st.write("You hav low chance of getting Brain Stroke")
