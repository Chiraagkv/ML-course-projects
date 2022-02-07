def train():    
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LinearRegression

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    np.random.seed(42)

    df = pd.read_csv('./data/data.csv')
    df.dropna(inplace=True)

    data = df.drop("Influencer", axis=1)
    data["Total"] = (data["TV"] + data["Radio"] + data["Social Media"])
    data["Profit"] = (data["Sales"] - data["Total"]) / data["Total"] * 100
    data["TV"] = data["TV"] / data["Total"] * 100
    data["Radio"] = data["Radio"] / data["Total"] * 100
    data["Social Media"] = data["Social Media"] / data["Total"] * 100

    X = data.drop(["TV", "Radio", "Social Media", "Sales"], axis=1)
    y = data.drop(["Profit", "Total"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    return model