import streamlit as st

st.sidebar.title('SIDE BAR')
option = st.sidebar.radio('Navigation', ['HOME', 'CURRENCY CONVERTER', 'SUPERVISED', 'UNSUPERVISED'])

def home():
    st.title('HOME')
    st.write("LET'S DRIVE INTO MACHINE LEARNING!")

    st.subheader('DEFINITION OF MACHINE LEARNING')
    st.write("Machine Learning is a subset of Artificial Intelligence (AI).")
    st.write("- It creates a predictive model.")
    st.write("- It trains a model on a dataset for predictive analysis.")

    st.subheader('TYPES OF MACHINE LEARNING')
    t = st.selectbox('SELECT ANY TYPE OF MACHINE LEARNING',options= ['TYPE','Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning'])

    if t == 'Supervised Learning':
        st.markdown('SUPERVISED LEARNING')

        st.subheader('DEFINITION')
        st.write("- Supervised Learning is a type of Machine Learning where the model learns from labeled data.")

        st.subheader('TECHNOLOGY')
        st.write('- Classification (0 or 1)')
        st.write('- Regression (Continuous Values)')

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Algorithm")
            st.write("- Linear Regression")
            st.write("- Logistic Regression")
            st.write("- Decision Tree")
            st.write("- Random Forest")

        with col2:
            st.markdown("Type")
            st.write("--> Regression")
            st.write("--> Classification")
            st.write("--> Both")
            st.write("--> Both")

    elif t == 'Unsupervised Learning':
        st.markdown('UNSUPERVISED LEARNING')

        st.subheader('DEFINITION')
        st.write("- Unsupervised Learning is a type of Machine Learning where the model learns from unlabeled data.")
        st.write("- After clustering we get unlabelled data(y feature/output) then it goesin supervised.")

        st.subheader('TECHNOLOGY')
        st.write('- Clustering')

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Algorithm")
            st.write("- K-Means Clustering")
            st.write("- DBSCAN")

        with col2:
            st.markdown("Type")
            st.write("--> Clustering")
            st.write("--> Clustering")

    elif t == 'Reinforcement Learning':
        st.markdown('REINFORCEMENT LEARNING')

        st.subheader('DEFINITION')
        st.write("- Reinforcement Learning is a type of Machine Learning where the model learns through feedback.")
        st.write("- It does not require any data.")

        st.subheader('TECHNOLOGY')
        st.write("- Model-Based RL")
        st.write("- Model-Free RL")

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Algorithm")
            st.write("- Q-Learning")
            st.write("- SARSA(State-Action-Reward-State-Action)")

        with col2:
            st.markdown("Type")
            st.write("--> Model-Free")
            st.write("--> Model-Free")

    st.subheader("SELECT AN ALGORITHM")
    alg = st.selectbox("Choose an algorithm", options=['CHOOSE',"Linear Regression","Logistic Regression","Decision Tree","Random Forest","K-Nearest Neighbors (KNN)","K-Means Clustering"])

    if alg == "Linear Regression":
        st.markdown('LINEAR REGRESSION')
        st.subheader('DEFINITION')
        st.write("- Linear Regression is the data which is continuous.")
        st.write("- It creates a Best Fit Line.")
        st.subheader('CODE')
        st.code('''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
ypred=lr.predict(x)
print(ypred)
        ''')
        st.subheader('GRAPH')
        st.image("linear.png")

    elif alg == "Logistic Regression":
        st.markdown('LOGISTIC REGRESSION')
        st.subheader('DEFINITION')
        st.write("Logistic Regression is a supervised learning algorithm used for classification problems.")
        st.write("It uses the sigmoid function. ")
        st.subheader("CODE")
        st.code('''
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(x, y)
ypred=lor.predict(x)
print(ypred)
        ''')
        st.subheader("GRAPH")
        st.image("logistic.png")

    elif alg == "Decision Tree":
        st.markdown("DECISION TREE")
        st.subheader('DEFINITION')
        st.write("Used for both classification and regression .")
        st.subheader("CODE")
        st.code('''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x, y)
ypred=dt.predict(x)
print(ypred)
        ''')
        st.subheader("GRAPH")
        st.image("decision.png")

    elif alg == "Random Forest":
        st.markdown("RANDOM FOREST")
        st.subheader('DEFINITION')
        st.write("It is an ensemble method that has multiple decision trees.")
        st.subheader("CODE")
        st.code('''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x, y)
ypred=rf.predict(x)
print(ypred)
        ''')
        st.subheader("GRAPH")
        # st.image("random.png")

    elif alg == "K-Nearest Neighbors (KNN)":
        st.markdown('K-NEAREST NEIGHBORS(KNN)')
        st.subheader('DEFINITION')
        st.write("- KNN can be used for both classification and clustering.")
        st.write("In unsupervised clustering (scatterplot), it: ")
        st.write("- Creates centroids.")
        st.write("- Groups the nearest data points to that centroid. The grouped points form a cluster.")
        st.subheader("CODE")
        st.code('''
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(x,y)
ypred=kn.predict(x)
print(ypred)
        ''')
        st.subheader("GRAPH")
        # st.image("images/knn.png")

    elif alg == "K-Means Clustering":
        st.markdown('K-MEANS CLUSTERING')
        st.write("- K-Means is an unsupervised clustering algorithm where K is the number of clusters,It works by:")
        st.write('- Grouping similar or nearest data points together.')
        st.write('- Finding the mean of those points.')
        st.write('- Assigning that mean to a cluster.')
        st.subheader("CODE")
        st.code('''
from sklearn.cluster import KMeans
inertia = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df)
    inertia.append(model.inertia_)
print(inertia)
        ''')
        st.subheader("GRAPH")
        # st.image("images/kmeans.png")

def currency_converter():
    st.title('CURRENCY CONVERTER')
    st.write('THIS IS THE CURRENCY CONVERTER PAGE.')

    from_ = st.selectbox("From Currency", options=['India ₹', 'United States $', 'euro €', 'Japanese Yen ¥', 'Russia ₽'])
    amount = st.number_input("Enter the amount: ")
    to = st.selectbox("To Currency", options=['India ₹', 'United States $', 'euro €', 'Japanese Yen ¥', 'Russia ₽'])
    btn = st.button("CONVERT")

    convert = {'India ₹': 1,'United States $': 83.33,'euro €': 91.20,'Japanese Yen ¥': 0.54,'Russia ₽': 0.77}

    if btn:
        try:
            if amount == 0.0:
                raise ValueError("Amount cannot be zero.")

            if from_ == to:
                st.success(f"{amount}  {from_} = {amount}  {to} (Same Currency)")
            else:
                c = amount * convert[from_]
                converted = c / convert[to]
                rate = convert[from_] / convert[to]
                reverse_rate = 1 / rate

                st.success(f"{amount} {from_} = {round(converted, 2)} {to}")
                st.markdown("CONVERSION INFO")
                st.info(f"1 {from_} = {round(rate, 4)} {to}")
                st.markdown("ADDITIONAL INFO")
                st.info(f"1 {to} = {round(reverse_rate, 4)} {from_}")

        except ValueError as e:
            st.error(f"{str(e)}")

        except KeyError:
            st.error("Selected currencies not available in the rate list.")
def supervised():
    st.title('SUPERVISED LEARNING')
    st.write('UPLOAD A CSV FILE TO PERFORM SUPERVISED ANALYSIS')

def unsupervised():
    st.title('UNSUPERVISED LEARNING')
    st.write('UPLOAD A CSV FILE TO PERFORM UNSUPERVISED ANALYSIS')

if option == "HOME":
    home()
elif option == "CURRENCY CONVERTER":
    currency_converter()
elif option == "SUPERVISED":
    supervised()
elif option == "UNSUPERVISED":
    unsupervised()
