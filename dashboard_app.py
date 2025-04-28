
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Function to generate true price based on rating
def true_price(rating):
    if rating <= 7.5:
        price = 0 + 6 * rating
    elif rating <= 8.5:
        price = 0 + 6 * rating + 20 * (rating - 7.5)**2
    else:
        price = 0 + 6 * rating + 20 * (1.0)**2 + 120 * (rating - 8.5)**4
    return price

# Generate dataset
def generate_data(n_samples=500, noise_std=2):
    ratings = np.clip(np.random.normal(7, 0.8, size=n_samples), 1, 10)
    prices = np.array([true_price(r) + np.random.normal(0, noise_std) for r in ratings])
    return ratings.reshape(-1, 1), prices

# Compute Bias², Variance, Noise, MSE
def compute_bias_variance(degree, n_experiments=100, n_points=100):
    x_test = np.linspace(1, 10, n_points).reshape(-1, 1)
    true_prices = np.array([true_price(x) for x in x_test.flatten()])
    predictions = np.zeros((n_experiments, n_points))
    
    for i in range(n_experiments):
        X_train, y_train = generate_data()
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        preds = model.predict(x_test)
        predictions[i, :] = preds
    
    avg_preds = np.mean(predictions, axis=0)
    bias_squared = (avg_preds - true_prices)**2
    variance = np.var(predictions, axis=0)
    noise = np.full_like(bias_squared, 4)  # noise_std^2 = 2^2 = 4
    mse = bias_squared + variance + noise

    return x_test.flatten(), bias_squared, variance, noise, mse

# Streamlit app
st.title("Bias-Variance Tradeoff in Predicting Footballer Prices")
st.markdown("""
### Story
We are modeling the relationship between a footballer's **season average rating** (out of 10) in **top leagues** (Premier League, La Liga, etc.) and their **market selling price**.

- Average players (rating 6–7.5): moderate prices (€30M–70M).
- Good players (rating 7.5–8.5): faster price growth.
- Top players (rating >8.5): prices explode (€150M+).

### What you see
This dashboard simulates multiple datasets and fits models of different complexity to predict player prices. We decompose the prediction error into:

- **Bias²**: systematic error.
- **Variance**: instability due to different training data.
- **Noise**: randomness we can't remove.
- **MSE**: Total mean squared error.

Try changing model complexity below!
""")

degree = st.slider("Choose model complexity (polynomial degree):", 1, 10, 1)

x_vals, bias2, var, noise, mse = compute_bias_variance(degree)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, mse, label="MSE (Total Error)", linewidth=2)
ax.plot(x_vals, bias2, label="Bias² (Error from wrong model)", linestyle='--')
ax.plot(x_vals, var, label="Variance (Error from instability)", linestyle='--')
ax.plot(x_vals, noise, label="Noise (Unavoidable error)", linestyle='--')
ax.set_xlabel("Average Rating (1-10)")
ax.set_ylabel("Error Components")
ax.set_title("MSE Decomposition by Model Complexity")
ax.legend()
ax.grid(True)

st.pyplot(fig)
