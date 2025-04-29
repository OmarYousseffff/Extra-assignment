
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
    noise = np.full_like(bias_squared, 4)  # because noise_std = 2
    mse = bias_squared + variance + noise

    return x_test.flatten(), bias_squared, variance, noise, mse

# Start Streamlit app
st.title("Bias-Variance Tradeoff in Predicting Footballer Prices")

# Full Explanation section before interaction
st.header("Introduction")
st.markdown(
    "In this dashboard, we simulate a real-world problem: predicting a football player's "
    "**market value** based on their **season-long average rating** (1–10) in a "
    "**top football league** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).

"
    "Players typically have ratings between 5 and 8.5, with exceptional stars rated higher. "
    "The market value increases non-linearly with ratings."
)

st.markdown("---")

st.header("Modeling Approach")
st.markdown(
    "- Ratings are generated from a normal distribution centered at 7.0.
"
    "- Prices grow:
"
    "  - Linearly up to 7.5
"
    "  - Quadratically between 7.5 and 8.5
"
    "  - Explosively (quartic) after 8.5
"
    "- Small random noise is added to simulate real-world randomness.

"
    "**Price Formula:**
"
    "- If rating ≤ 7.5:  Price = 6 × rating
"
    "- If 7.5 < rating ≤ 8.5:  Price = 6 × rating + 20 × (rating - 7.5)²
"
    "- If rating > 8.5:  Price = 6 × rating + 20 + 120 × (rating - 8.5)^4"
)

st.markdown("---")

st.header("Bias-Variance Decomposition")
st.markdown(
    "- **Bias²** measures how much a model's average prediction misses the true price.
"
    "- **Variance** measures how much predictions vary if training data changes.
"
    "- **Noise** is the unavoidable randomness in data.

"
    "The total prediction error (MSE) is decomposed as:
"
    "**MSE = Bias² + Variance + Noise**"
)

st.markdown("---")

st.header("Real Football Examples")
st.markdown(
    "- **Bias²** example: Simple models (like a straight line) **underestimate** prices "
    "of superstars like Messi, Mbappé, Haaland.
"
    "- **Variance** example: Overly complex models (high-degree polynomials) **overreact** "
    "to small data changes.
"
    "- **Noise** example: Unpredictable real-world events (injuries, negotiations) affect "
    "prices and create random error.

"
    "Understanding Bias² and Variance helps explain why model complexity matters."
)

st.markdown("---")

st.header("How to Use the Dashboard")
st.markdown(
    "- Adjust the **polynomial degree** using the slider below.
"
    "- Watch how **Bias²**, **Variance**, **Noise**, and **Total MSE** change.
"
    "- Find the **best complexity** where MSE is minimized!"
)

st.markdown("---")

# Now interactive part
st.subheader("Choose model complexity:")
degree = st.slider("Polynomial Degree", 1, 10, 1)

# Plot
x_vals, bias2, var, noise, mse = compute_bias_variance(degree)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, mse, label="MSE (Total Error)", linewidth=2)
ax.plot(x_vals, bias2, label="Bias² (Systematic error)", linestyle='--')
ax.plot(x_vals, var, label="Variance (Instability)", linestyle='--')
ax.plot(x_vals, noise, label="Noise (Unpredictable randomness)", linestyle='--')
ax.set_xlabel("Average Rating (1-10)")
ax.set_ylabel("Error Components")
ax.set_title("MSE Decomposition by Model Complexity")
ax.legend()
ax.grid(True)

st.pyplot(fig)
