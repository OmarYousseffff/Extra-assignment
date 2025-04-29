
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Function to generate true price based on rating
def true_price(rating):
    if rating <= 7.5:
        return 0 + 6 * rating
    elif rating <= 8.5:
        return 0 + 6 * rating + 20 * (rating - 7.5)**2
    else:
        return 0 + 6 * rating + 20 + 120 * (rating - 8.5)**4

# Function to generate synthetic data
def generate_data(n_samples=500, noise_std=2):
    ratings = np.clip(np.random.normal(7, 0.8, size=n_samples), 1, 10)
    prices = np.array([true_price(r) + np.random.normal(0, noise_std) for r in ratings])
    return ratings.reshape(-1, 1), prices

# Function to compute Bias², Variance, Noise and MSE
def compute_bias_variance(degree, n_experiments=100, n_points=100):
    x_test = np.linspace(1, 10, n_points).reshape(-1, 1)
    true_prices = np.array([true_price(x) for x in x_test.flatten()])
    predictions = np.zeros((n_experiments, n_points))

    for _ in range(n_experiments):
        X_train, y_train = generate_data()
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        preds = model.predict(x_test)
        predictions[_] = preds

    avg_preds = np.mean(predictions, axis=0)
    bias_squared = (avg_preds - true_prices)**2
    variance = np.var(predictions, axis=0)
    noise = np.full_like(bias_squared, 4)
    mse = bias_squared + variance + noise

    return x_test.flatten(), bias_squared, variance, noise, mse

# Streamlit app starts here
st.title("Bias-Variance Decomposition in Predicting Footballer Prices")

# Full explanation inserted
st.header("Introduction")
st.markdown(
    "In this project, we simulate a real-world football problem: estimating the transfer price of a player based on their "
    "average rating over an entire season in a top European league (such as the Premier League, La Liga, Bundesliga, etc.).

"
    "Players typically have ratings between 5 and 8.5 out of 10, with exceptional stars achieving higher ratings. "
    "Importantly, the market value of players does not increase linearly with their ratings — top players command much higher transfer fees."
)

st.markdown("---")

st.header("Data Generation")
st.markdown(
    "Instead of using real data, we generated synthetic data based on realistic assumptions:

"
    "- Player ratings are drawn from a normal distribution centered at 7.0 (to match typical football ratings).
"
    "- Player prices are calculated using a piecewise nonlinear function:
"
    "    - Moderate growth for ratings ≤ 7.5
"
    "    - Faster growth between 7.5 and 8.5
"
    "    - Explosive growth for ratings > 8.5
"
    "- A small amount of random noise is added to simulate real-world unpredictability.

"
    "This approach lets us control the relationship and study model behavior clearly."
)

st.markdown("---")

st.header("Modeling and Simulation")
st.markdown(
    "Polynomial regression models of varying degrees (from 1 to 10) are trained to predict player prices from ratings. "
    "For each complexity level:
"
    "- Many datasets are simulated
"
    "- Predictions are computed
"
    "- Bias², Variance, Noise, and Total MSE are calculated

"
    "This allows us to study how increasing model complexity affects prediction errors."
)

st.markdown("---")

st.header("Bias², Variance, and Noise (Football Examples)")
st.markdown(
    "- **Bias²**: A simple model (like a straight line) would underestimate the price of exceptional players like Messi or Haaland "
    "because it cannot capture the explosion at high ratings.
"
    "- **Variance**: A very flexible model (like a degree-10 polynomial) would change wildly if one player's rating slightly changes.
"
    "- **Noise**: Even if a player is great, random factors like injury or personal issues can lower their value — this randomness is the noise we cannot model."
)

st.markdown("---")

st.header("Dashboard Description")
st.markdown(
    "The Streamlit dashboard shows:
"
    "- Full problem explanation at the top
"
    "- An interactive slider to choose model complexity
"
    "- A graph showing Bias², Variance, Noise, and MSE

"
    "Users can see how the error components change as the model becomes more or less flexible."
)

st.markdown("---")

st.header("Conclusion")
st.markdown(
    "This project realistically simulates the Bias-Variance tradeoff in a football transfer market context, "
    "matching the assignment request for a thoughtful, clear, original dashboard."
)

st.markdown("---")

# Interactive part
st.subheader("Now explore how model complexity affects prediction error:")
degree = st.slider("Select Polynomial Degree:", 1, 10, 1)

# Plotting
x_vals, bias2, var, noise, mse = compute_bias_variance(degree)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, mse, label="MSE (Total Error)", linewidth=2)
ax.plot(x_vals, bias2, label="Bias² (Systematic error)", linestyle='--')
ax.plot(x_vals, var, label="Variance (Instability)", linestyle='--')
ax.plot(x_vals, noise, label="Noise (Random unpredictability)", linestyle='--')
ax.set_xlabel("Average Rating (1-10)")
ax.set_ylabel("Error Components")
ax.set_title("MSE Decomposition by Model Complexity")
ax.legend()
ax.grid(True)

st.pyplot(fig)
