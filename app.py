import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

@st.cache_resource
def train_model(df):
    # Encode categorical variables
    le = LabelEncoder()
    df_processed = df.copy()
    df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
    df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])

    # Split features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

# Streamlit app
st.title('Titanic Survival Prediction')
st.write('Predict whether a passenger would survive the Titanic disaster based on their characteristics.')

# Load data and train model
df = load_data()
model, feature_names = train_model(df)

# Sidebar for user input
st.sidebar.header('Passenger Information')

# Input fields
pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0, 100, 25)
sibsp = st.sidebar.slider('Number of Siblings/Spouses', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children', 0, 6, 0)
fare = st.sidebar.slider('Fare', 0.0, 512.0, 32.0)
embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encode inputs
sex_encoded = 1 if sex == 'male' else 0
embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Create input DataFrame with proper column names
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Make prediction
if st.sidebar.button('Predict Survival'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.header('Prediction Result')
    if prediction == 1:
        st.success(f'The passenger would likely survive! (Probability: {probability:.2f})')
    else:
        st.error(f'The passenger would likely not survive. (Probability: {probability:.2f})')

# Display dataset info
st.header('Dataset Overview')
st.write(f'Dataset shape: {df.shape}')
st.write('First 5 rows:')
st.dataframe(df.head())

# Feature importance
st.header('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
plt.title('Feature Importance')
st.pyplot(fig)

# Survival statistics
st.header('Survival Statistics')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Survival Rate by Class')
    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    st.bar_chart(survival_by_class)

with col2:
    st.subheader('Survival Rate by Sex')
    survival_by_sex = df.groupby('Sex')['Survived'].mean()
    st.bar_chart(survival_by_sex)
