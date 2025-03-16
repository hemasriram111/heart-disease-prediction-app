# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\hemas\Desktop\heart\heart.csv")

# HEADINGS
st.title('Heart Disease Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['target'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.slider('Sex (0 = Female, 1 = Male)', 0, 1, 1)
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 200)
    fbs = st.sidebar.slider('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', 0, 1, 0)
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (0-2)', 0, 2, 0)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.slider('Exercise Induced Angina (1 = Yes, 0 = No)', 0, 1, 0)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)', 0, 2, 1)

    user_report_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0] == 0:
    color = 'blue'
    output = 'You are Healthy'
else:
    color = 'red'
    output = 'You have Heart Disease'

# Age vs Max Heart Rate
st.header('Age vs Max Heart Rate Graph (Others vs Yours)')
fig_thalach = plt.figure()
ax1 = sns.scatterplot(x='age', y='thalach', data=df, hue='target', palette='Greens')
ax2 = sns.scatterplot(x=user_data['age'], y=user_data['thalach'], s=150, color=color)
plt.xticks(np.arange(30, 80, 5))
plt.yticks(np.arange(70, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_thalach)

# Age vs Cholesterol
st.header('Age vs Cholesterol Graph (Others vs Yours)')
fig_chol = plt.figure()
ax3 = sns.scatterplot(x='age', y='chol', data=df, hue='target', palette='magma')
ax4 = sns.scatterplot(x=user_data['age'], y=user_data['chol'], s=150, color=color)
plt.xticks(np.arange(30, 80, 5))
plt.yticks(np.arange(100, 600, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_chol)

# Age vs Blood Pressure
st.header('Age vs Blood Pressure Graph (Others vs Yours)')
fig_trestbps = plt.figure()
ax5 = sns.scatterplot(x='age', y='trestbps', data=df, hue='target', palette='Reds')
ax6 = sns.scatterplot(x=user_data['age'], y=user_data['trestbps'], s=150, color=color)
plt.xticks(np.arange(30, 80, 5))
plt.yticks(np.arange(90, 210, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_trestbps)

# Age vs ST Depression
st.header('Age vs ST Depression Graph (Others vs Yours)')
fig_oldpeak = plt.figure()
ax7 = sns.scatterplot(x='age', y='oldpeak', data=df, hue='target', palette='Blues')
ax8 = sns.scatterplot(x=user_data['age'], y=user_data['oldpeak'], s=150, color=color)
plt.xticks(np.arange(30, 80, 5))
plt.yticks(np.arange(0, 7, 0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_oldpeak)

# OUTPUT
st.subheader('Your Report: ')
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')