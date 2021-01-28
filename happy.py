import streamlit as st
import pandas as pd
import numpy as np
import plotly as dd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
 
st.title('World Happiness 2019')
 
 
df = pd.read_csv("Happy.csv")
 
if st.checkbox('Show dataframe'):
    st.write(df)
 
st.subheader('Scatterplot')
 
yy= st.selectbox('Select the variable for the y-axis of your scatterplot', df.columns)
xx = st.selectbox('Select the variable for the x-axis of your scatterplot?', df.columns)
 # create figure using plotly express
fig = px.scatter(df, x =xx,y=yy)
st.plotly_chart(fig)
 
st.subheader('Histogram')
 
feature = st.selectbox('For which variable you would like to draw a histogram?', df.columns[0:6])
# Filter dataframe
fig2 = px.histogram(df, x=feature, marginal="rug")
st.plotly_chart(fig2)
 
st.subheader('Machine Learning models')

y_var= st.selectbox('Your target variable?', df.columns)
dfnew=df.drop(y_var, axis=1)
x_var = st.multiselect('Your independent variable/variables? (you can select multiple variables) ', dfnew.columns)
 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

xdata=df[x_var]
#st.write (xdata)
ydata=df[y_var]
#st.write (ydata)

X_train,X_test, y_train, y_test = train_test_split(xdata,ydata, train_size=0.7, random_state=1)

alg = ['Multiple Linear Regression','Neural Networks']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Multiple Linear Regression':
    
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    st.write('Coefficients: \n', regr.coef_)
    st.write('Mean squared error: %.2f'
    % mean_squared_error(y_test, y_pred))
    st.write('R2',  r2_score(y_test, y_pred))
    
elif classifier=='Neural Networks':
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    regr.predict(X_test[:2])
    acc=regr.score(X_test, y_test)
    st.write('Accuracy: ', acc)
  
     
#elif classifier == 'Support Vector Machine':
#    svm=SVC()
#    svm.fit(X_train, y_train)
#    acc = svm.score(X_test, y_test)
#    st.write('Accuracy: ', acc)
 #   pred_svm = svm.predict(X_test)
 #   cm=confusion_matrix(y_test,pred_svm)
 #   st.write('Confusion matrix: ', cm)