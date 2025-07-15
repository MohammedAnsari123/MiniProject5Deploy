import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

st.sidebar.title("Side Bar")
option = st.sidebar.radio("",["Home","Currency_Converter","Supervised","Unsupervised"])

def home():
    st.title("Machine Learning Algorithms")

    st.subheader("Machine Learning Defination")
    st.write("Machine Learning is a subset of AI It creates a predictive model. It trains a model on a dataset for predictive analysis.")

    ml_types = st.selectbox("Types of ML", options = ['', 'Supervised','Unsupervised','Reinforcement'])

    if ml_types == "Supervised":
        st.write("Supervised :- Supervised Learning is a type of Machine Learning where the model learns from labeled data.")

        st.subheader("Technologies :-")
        st.write('- Classification (0 or 1)')
        st.write('- Regression (Continuous Values)')

        st.subheader("Algorithm :-")
        st.write("Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, K-NN, Neural Networks")
        st.write("Linear Regression, Ridge/Lasso Regression, Decision Tree, Random Forest Regressor, SVR")

    elif ml_types == "Unsupervised":
        st.write("Unsupervised :- Finds patterns in unlabeled data")

        st.subheader("Technologies :-")
        st.write('- Clustering')

        st.subheader("Algorithm :-")
        st.write("K-Means, DBSCAN")

    elif ml_types == "Reinforcement":
        st.write("Reinforcement :- Learn by train and error")
        st.subheader("Technologies :-")
        st.write("- Model-Based RL")
        st.write("- Model-Free RL")

        st.subheader("Algorithm :-")
        st.write("- Q-Learning")
        st.write("- SARSA(State-Action-Reward-State-Action)")

    ml_algo = st.selectbox("ML Algorithm", options = ["", "Linear", "Logistic", "Random Forest", "Decision Tree", "KNN", "KMeans"])

    if ml_algo == "Linear":
        st.write("Continuous flow of data (y = mx + b) linear equation")
        st.image("Linear.png")
        st.code("""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
ypred = lr.predict(x)
""", language='python')

    elif ml_algo == "Logistic":
        st.write("In Logistic Regression, sigmoid function is udes range of the functions is 0 to 1 f(X) = (1, 0)")
        st.image("logistic.jpg")
        st.code("""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x, y)
ypred = lr.predict(X)
""", language='python')

    elif ml_algo == "Random Forest":
        st.write("Multiple Decision Tree")
        st.image("RF.png")
        st.code("""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x, y)
ypred = rf.predict(x)
""", language='python')

    elif ml_algo == "Decision Tree":
        st.write("")
        st.image("DT.png")
        st.code("""
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x, y)
ypred = dt.predict(x)
""", language='python')

    elif ml_algo == "KNN":
        st.write("K Nearest Neighbors")
        st.image("KNN.png")
        st.code("""
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(x, y)
ypred = kn.predict(x)
""", language='python')

    elif ml_algo == "KMeans":
        st.write("Create a small small group of data")
        st.image("KMeans.png")
        st.code('''
from sklearn.cluster import KMeans
inertia = []
for k in range(1, 10):
model = KMeans(n_clusters=k, random_state=42)
model.fit(df)
inertia.append(model.inertia_)
print(inertia)
        ''')


def Currency_Converter():
    currop1 = st.selectbox("Check Currency1", options = ['India ₹','United States $','euro €','Japanese Yen ¥','Russia ₽'])
    amount = st.number_input("Enter The Amount")
    currop2 = st.selectbox("Check Currency2", options = ['India ₹','United States $','euro €','Japanese Yen ¥','Russia ₽'])

    Currency_Converter_button = st.button("Convert the currency")

    conversion_rates = {
    ('India ₹', 'United States $'): 0.012,
    ('India ₹', 'euro €'): 0.011,
    ('India ₹', 'Japanese Yen ¥'): 1.86,
    ('India ₹', 'Russia ₽'): 1.30,

    ('United States $', 'India ₹'): 83.33,
    ('United States $', 'euro €'): 0.92,
    ('United States $', 'Japanese Yen ¥'): 156.80,
    ('United States $', 'Russia ₽'): 108.50,

    ('euro €', 'India ₹'): 91.20,
    ('euro €', 'United States $'): 1.09,
    ('euro €', 'Japanese Yen ¥'): 170.40,
    ('euro €', 'Russia ₽'): 117.90,

    ('Japanese Yen ¥', 'India ₹'): 0.54,
    ('Japanese Yen ¥', 'United States $'): 0.0064,
    ('Japanese Yen ¥', 'euro €'): 0.0059,
    ('Japanese Yen ¥', 'Russia ₽'): 0.69,

    ('Russia ₽', 'India ₹'): 0.77,
    ('Russia ₽', 'United States $'): 0.0092,
    ('Russia ₽', 'euro €'): 0.0085,
    ('Russia ₽', 'Japanese Yen ¥'): 1.45,
}

    if Currency_Converter_button:
        if currop1 == currop2:
            st.write(f"{amount} {currop1} is equal to {amount} {currop2}")


        else:
            rate = conversion_rates.get((currop1, currop2))
            if rate:
                result = amount * rate
                st.write(f"{amount} {currop1} is equal to {int(result * 100)/100} {currop2}")
            else:
                st.warning("Conversion rate not available for selected currencies.")


def Supervised():
    df = pd.read_csv("Salary_dataset.csv")

    SupSalaryDIC = st.selectbox("", options = ["Describe", "Info", "Columns"])
    if SupSalaryDIC == "Describe":
        st.write(df.describe())
    elif SupSalaryDIC == "Info":
        st.write(df.info())
    elif SupSalaryDIC == "Columns":
        st.write(df.columns)





def Unsupervised():
    df = pd.read_csv("Mall_Customers.csv")









if option == "Home":
    home()

elif option == "Currency_Converter":
    Currency_Converter()

elif option == "Supervised":
    Supervised()

elif option == "Unsupervised":
    Unsupervised()



