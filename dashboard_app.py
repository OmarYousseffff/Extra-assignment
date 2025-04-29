
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
        price = 0 + 6 * rating + 20 + 120 * (rating - 8.5)**4
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
## Introduction
In this dashboard, we model a real-world challenge: predicting a football player's **market value** based on their **season-long average performance rating** (out of 10) in a **top football league** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).

Players are typically rated between **5 and 8.5**:
- Average players (5–7.5): moderate value
- Good players (7.5–8.5): higher value
- Exceptional players (>8.5): extremely rare and expensive

The relationship between **rating and price** is **non-linear**:
- Prices grow moderately up to rating 7.5
- Grow faster between 7.5 and 8.5
- Explode after 8.5 (elite players)

## Modeling Approach
Ratings are drawn from a normal distribution (centered at 7.0, clipped between 1 and 10).  
Prices are determined by:
- Linear growth up to 7.5
- Quadratic growth from 7.5 to 8.5
- Quartic explosion above 8.5

Noise (small random fluctuation) is added to simulate real-world uncertainty.

**Price Formula:**
- If rating ≤ 7.5: Price = 6 × rating
- If 7.5 < rating ≤ 8.5: Price = 6 × rating + 20 × (rating - 7.5)^2
- If rating > 8.5: Price = 6 × rating + 20 + 120 × (rating - 8.5)^4

## Bias-Variance Decomposition
Predictive models face a tradeoff:
- **Simple models** (low degree polynomials): High Bias, Low Variance
- **Complex models** (high degree polynomials): Low Bias, High Variance

The total prediction error (MSE) decomposes into:
- **Bias²**: average wrongness
- **Variance**: instability due to sample variation
- **Noise**: irreducible randomness

**MSE = Bias² + Variance + Noise**

## Bias², Variance, and Noise Explained (with Football Examples)

**Bias² (Systematic Error):**  
Suppose we use a **very simple model** (a straight line) to predict prices.  
It will **miss** the fact that top players (ratings >8.5) become **much more expensive**.  
Thus, it **underestimates** the prices of stars like Messi, Mbappé, or Haaland — leading to **high Bias²**.

**Variance (Instability):**  
Suppose we use a **very complex model** (like a 10th-degree polynomial).  
Small changes in training data — like one extra player rated 8.3 instead of 8.2 — can **change the predictions a lot**.  
This makes the model **unstable** and causes **high Variance**.

**Noise (Randomness):**  
Even if a player has a high rating, external factors (injury, contract issues) can **lower their transfer value unexpectedly**.  
This random effect is **Noise**, which **no model can perfectly predict**.

## How to Use the Dashboard
Use the slider below to adjust model complexity (polynomial degree) and observe:
- How Bias² decreases with complexity
- How Variance increases with complexity
- How MSE finds a minimum at a good model complexity
""")

degree = st.slider("Choose model complexity (polynomial degree):", 1, 10, 1)

x_vals, bias2, var, noise, mse = compute_bias_variance(degree)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, mse, label="MSE (Total Error)", linewidth=2)
ax.plot(x_vals, bias2, label="Bias² (Systematic error)", linestyle='--')
ax.plot(x_vals, var, label="Variance (Instability)", linestyle='--')
ax.plot(x_vals, noise, label="Noise (Randomness)", linestyle='--')
ax.set_xlabel("Average Rating (1-10)")
ax.set_ylabel("Error Components")
ax.set_title("MSE Decomposition by Model Complexity")
ax.legend()
ax.grid(True)

st.pyplot(fig)
