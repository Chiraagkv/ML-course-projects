import streamlit as st
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from backend import train
np.random.seed(42)

st.title("Budget Distributor")

total = float(st.text_input("Enter Budget Available for Advertising", "23.34"))
sales = float(st.text_input("Enter expected profit percentage", "100"))
inputs = np.array([[total, sales]])

regressor = train()

predictions= regressor.predict(inputs)

# plotting
try:
    from matplotlib import pyplot as plt
    import numpy as np
    fig = plt.figure()
    plt.suptitle("Budget Distribution Across Different Media")
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    media = ['TV', 'Radio', 'Social Media']
    ax.pie(predictions[0, :3], labels=media, autopct='%1.2f%%')
    st.pyplot(fig)
except ValueError:
    st.write('Your numbers are HUGE! try something smaller')