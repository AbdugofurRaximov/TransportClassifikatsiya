import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

# title
st.title("Transport klasifikatsiya qiladigan model")

#  rasmni joylash
file=st.file_uploader("Rasm yuklash", type=['png','jpeg','gif','svg'])
st.image(file)
# PIL convet

img = PILImage.create(file)  
# model
model=load_learner("transport_model.pkl")

# prediction
pred,pred_id,prob = model.predict(img)
st.success(f"Bashorat: {pred}")
st.info(f"Ehtimollik: {prob[pred_id]*100:.1f}%")


#  plotting 
fig = px.bar(x=prob*100,y=model.dls.vocab)
st.plotly_chart(fig)