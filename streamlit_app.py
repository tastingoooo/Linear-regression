"""
Interactive Simple Linear Regression with Streamlit
CRISP-DM Methodology Implementation with User Controls

Features:
- Adjustable slope parameter (a in ax+b)
- Adjustable noise level
- Adjustable number of data points
- Real-time visualization
- Interactive model evaluation
"""

import streamlit as st
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Interactive Linear Regression - CRISP-DM",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.step-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏠 Interactive Linear Regression Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">CRISP-DM Methodology with Real-time Controls</h2>', unsafe_allow_html=True)

# Sidebar for user controls
st.sidebar.header("🎛️ Model Parameters")
st.sidebar.markdown("---")

# Step 1: Business Understanding
st.markdown('<h2 class="step-header">📈 Step 1: Business Understanding</h2>', unsafe_allow_html=True)
st.info("""
**Business Objective**: Build an interactive linear regression model to predict house prices  
**Analysis Goal**: Explore how different parameters affect the linear relationship  
**Success Criteria**: R² value > 0.7, minimize RMSE  
**Interactive Features**: Real-time parameter adjustment and visualization
""")

# User input controls in sidebar
st.sidebar.subheader("🔧 Data Generation Parameters")
slope_param = st.sidebar.slider(
    "Slope (a in ax+b)", 
    min_value=0.5, 
    max_value=5.0, 
    value=2.5, 
    step=0.1,
    help="Controls how much price increases per square meter"
)

noise_level = st.sidebar.slider(
    "Noise Level", 
    min_value=0.0, 
    max_value=50.0, 
    value=20.0, 
    step=2.5,
    help="Amount of random variation in the data"
)

n_samples = st.sidebar.slider(
    "Number of Data Points", 
    min_value=50, 
    max_value=500, 
    value=100, 
    step=25,
    help="Total number of house samples to generate"
)

intercept_param = st.sidebar.slider(
    "Intercept (b in ax+b)", 
    min_value=20.0, 
    max_value=100.0, 
    value=50.0, 
    step=5.0,
    help="Base price when area is 0"
)

# Advanced parameters
st.sidebar.subheader("🎯 Advanced Settings")
test_size = st.sidebar.slider(
    "Test Set Ratio", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.2, 
    step=0.05,
    help="Proportion of data used for testing"
)

random_seed = st.sidebar.number_input(
    "Random Seed", 
    min_value=1, 
    max_value=1000, 
    value=42,
    help="For reproducible results"
)

# Generate data based on user inputs
@st.cache_data
def generate_data(slope, noise, n_points, intercept, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    house_sizes = []
    prices = []
    
    for i in range(n_points):
        # Generate house area (50-250 square meters)
        size = random.normalvariate(120, 30)
        size = max(50, min(250, size))
        
        # Generate house price using user-defined parameters
        price = slope * size + random.normalvariate(0, noise) + intercept
        
        house_sizes.append(size)
        prices.append(price)
    
    return house_sizes, prices

# Data processing functions
def calculate_stats(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(variance)
    sorted_data = sorted(data)
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    median = sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    
    return {
        'count': n,
        'mean': mean,
        'std': std,
        'min': minimum,
        'median': median,
        'max': maximum
    }

def correlation_coefficient(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    return numerator / denominator if denominator != 0 else 0

def detect_outliers(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers, lower_bound, upper_bound

def linear_regression(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0
    
    intercept = mean_y - slope * mean_x
    
    return slope, intercept

def predict(x, slope, intercept):
    return slope * x + intercept

def r_squared(y_actual, y_predicted):
    mean_y = sum(y_actual) / len(y_actual)
    ss_tot = sum((y - mean_y) ** 2 for y in y_actual)
    ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def rmse(y_actual, y_predicted):
    mse = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual))) / len(y_actual)
    return math.sqrt(mse)

def mae(y_actual, y_predicted):
    return sum(abs(y_actual[i] - y_predicted[i]) for i in range(len(y_actual))) / len(y_actual)

# Generate data
house_sizes, prices = generate_data(slope_param, noise_level, n_samples, intercept_param, random_seed)

# Step 2: Data Understanding
st.markdown('<h2 class="step-header">📊 Step 2: Data Understanding</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Samples", n_samples)
with col2:
    st.metric("Features", 1)
with col3:
    st.metric("True Slope", f"{slope_param:.2f}")
with col4:
    st.metric("Noise Level", f"{noise_level:.1f}")

# Calculate statistics
size_stats = calculate_stats(house_sizes)
price_stats = calculate_stats(prices)
correlation = correlation_coefficient(house_sizes, prices)

# Display statistics in columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 House Area Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Median', 'Max'],
        'Value': [
            f"{size_stats['mean']:.2f} sqm",
            f"{size_stats['std']:.2f}",
            f"{size_stats['min']:.2f}",
            f"{size_stats['median']:.2f}",
            f"{size_stats['max']:.2f}"
        ]
    })
    st.dataframe(stats_df, use_container_width=True)

with col2:
    st.subheader("💰 House Price Statistics")
    price_stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Median', 'Max'],
        'Value': [
            f"{price_stats['mean']:.2f} 10K CNY",
            f"{price_stats['std']:.2f}",
            f"{price_stats['min']:.2f}",
            f"{price_stats['median']:.2f}",
            f"{price_stats['max']:.2f}"
        ]
    })
    st.dataframe(price_stats_df, use_container_width=True)

# Correlation
st.markdown(f"""
<div class="metric-box">
<h4>🔗 Correlation Coefficient: {correlation:.4f}</h4>
<p>Shows the strength of linear relationship between area and price.</p>
</div>
""", unsafe_allow_html=True)

# Data visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot
ax1.scatter(house_sizes, prices, alpha=0.6, color='blue')
ax1.set_xlabel('House Area (sqm)')
ax1.set_ylabel('House Price (10K CNY)')
ax1.set_title('Area vs Price Scatter Plot')
ax1.grid(True, alpha=0.3)

# Area distribution
ax2.hist(house_sizes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
ax2.set_xlabel('House Area (sqm)')
ax2.set_ylabel('Frequency')
ax2.set_title('House Area Distribution')
ax2.grid(True, alpha=0.3)

# Price distribution
ax3.hist(prices, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
ax3.set_xlabel('House Price (10K CNY)')
ax3.set_ylabel('Frequency')
ax3.set_title('House Price Distribution')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Step 3: Data Preparation
st.markdown('<h2 class="step-header">🔧 Step 3: Data Preparation</h2>', unsafe_allow_html=True)

# Outlier detection
size_outliers, size_lb, size_ub = detect_outliers(house_sizes)
price_outliers, price_lb, price_ub = detect_outliers(prices)

col1, col2 = st.columns(2)
with col1:
    st.metric("Area Outliers", len(size_outliers))
    st.caption(f"Normal range: [{size_lb:.2f}, {size_ub:.2f}]")

with col2:
    st.metric("Price Outliers", len(price_outliers))
    st.caption(f"Normal range: [{price_lb:.2f}, {price_ub:.2f}]")

# Clean data
clean_sizes = []
clean_prices = []

for i in range(len(house_sizes)):
    if (size_lb <= house_sizes[i] <= size_ub and 
        price_lb <= prices[i] <= price_ub):
        clean_sizes.append(house_sizes[i])
        clean_prices.append(prices[i])

# Train-test split
train_size = int((1 - test_size) * len(clean_sizes))
combined = list(zip(clean_sizes, clean_prices))
random.shuffle(combined)

train_data = combined[:train_size]
test_data = combined[train_size:]

x_train = [item[0] for item in train_data]
y_train = [item[1] for item in train_data]
x_test = [item[0] for item in test_data]
y_test = [item[1] for item in test_data]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cleaned Samples", len(clean_sizes))
with col2:
    st.metric("Training Set", len(x_train))
with col3:
    st.metric("Test Set", len(x_test))

# Step 4: Modeling
st.markdown('<h2 class="step-header">🤖 Step 4: Modeling</h2>', unsafe_allow_html=True)

# Train model
slope, intercept = linear_regression(x_train, y_train)

col1, col2 = st.columns(2)
with col1:
    st.metric("Learned Slope", f"{slope:.4f}")
    st.caption("Predicted relationship")
with col2:
    st.metric("Learned Intercept", f"{intercept:.4f}")
    st.caption("Base price")

st.markdown(f"""
<div class="metric-box">
<h4>📐 Regression Equation</h4>
<p><strong>Price = {slope:.4f} × Area + {intercept:.4f}</strong></p>
<p>Compare with true equation: Price = {slope_param:.4f} × Area + {intercept_param:.4f}</p>
</div>
""", unsafe_allow_html=True)

# Make predictions
y_train_pred = [predict(x, slope, intercept) for x in x_train]
y_test_pred = [predict(x, slope, intercept) for x in x_test]

# Step 5: Evaluation
st.markdown('<h2 class="step-header">📋 Step 5: Evaluation</h2>', unsafe_allow_html=True)

# Calculate metrics
train_r2 = r_squared(y_train, y_train_pred)
test_r2 = r_squared(y_test, y_test_pred)
train_rmse = rmse(y_train, y_train_pred)
test_rmse = rmse(y_test, y_test_pred)
train_mae = mae(y_train, y_train_pred)
test_mae = mae(y_test, y_test_pred)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training R²", f"{train_r2:.4f}")
    st.metric("Test R²", f"{test_r2:.4f}")

with col2:
    st.metric("Training RMSE", f"{train_rmse:.2f}")
    st.metric("Test RMSE", f"{test_rmse:.2f}")

with col3:
    st.metric("Training MAE", f"{train_mae:.2f}")
    st.metric("Test MAE", f"{test_mae:.2f}")

# Model validation
performance_status = "✅ Excellent" if test_r2 > 0.9 else "✅ Good" if test_r2 > 0.7 else "⚠️ Needs Improvement"
generalization_status = "✅ Good" if abs(train_r2 - test_r2) < 0.1 else "⚠️ Potential Overfitting"

st.markdown(f"""
<div class="metric-box">
<h4>🎯 Model Validation</h4>
<p><strong>Performance:</strong> {performance_status} (R² = {test_r2:.4f})</p>
<p><strong>Generalization:</strong> {generalization_status} (Difference: {abs(train_r2 - test_r2):.4f})</p>
</div>
""", unsafe_allow_html=True)

# Visualization of results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Actual vs Predicted
ax1.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
ax1.set_xlabel('Actual Price')
ax1.set_ylabel('Predicted Price')
ax1.set_title('Actual vs Predicted')
ax1.grid(True, alpha=0.3)

# Residuals plot
residuals = [y_test[i] - y_test_pred[i] for i in range(len(y_test))]
ax2.scatter(y_test_pred, residuals, alpha=0.6, color='green')
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Price')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals Plot')
ax2.grid(True, alpha=0.3)

# Regression line
ax3.scatter(x_test, y_test, alpha=0.6, color='blue', label='Actual Data')
x_range = np.linspace(min(x_test), max(x_test), 100)
y_range_pred = [predict(x, slope, intercept) for x in x_range]
ax3.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
ax3.set_xlabel('House Area (sqm)')
ax3.set_ylabel('House Price (10K CNY)')
ax3.set_title('Linear Regression Fit')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Residuals histogram
ax4.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Residuals Distribution')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Prediction results table
st.subheader("🔍 Sample Predictions")
if len(x_test) > 0:
    sample_size = min(10, len(x_test))
    predictions_df = pd.DataFrame({
        'Area (sqm)': [f"{x_test[i]:.1f}" for i in range(sample_size)],
        'Actual Price': [f"{y_test[i]:.2f}" for i in range(sample_size)],
        'Predicted Price': [f"{y_test_pred[i]:.2f}" for i in range(sample_size)],
        'Error': [f"{abs(y_test[i] - y_test_pred[i]):.2f}" for i in range(sample_size)]
    })
    st.dataframe(predictions_df, use_container_width=True)

# Step 6: Deployment
st.markdown('<h2 class="step-header">🚀 Step 6: Deployment</h2>', unsafe_allow_html=True)

st.subheader("🏠 House Price Predictor")
col1, col2 = st.columns(2)

with col1:
    user_area = st.number_input(
        "Enter house area (sqm):", 
        min_value=50.0, 
        max_value=300.0, 
        value=120.0, 
        step=5.0
    )

with col2:
    predicted_price = predict(user_area, slope, intercept)
    st.metric("Predicted Price", f"{predicted_price:.2f} 10K CNY")

# Batch prediction examples
st.subheader("📊 Example Predictions")
example_areas = [80, 100, 120, 150, 200]
example_predictions = []

for area in example_areas:
    price = predict(area, slope, intercept)
    example_predictions.append({
        'Area (sqm)': area,
        'Predicted Price (10K CNY)': f"{price:.2f}"
    })

examples_df = pd.DataFrame(example_predictions)
st.dataframe(examples_df, use_container_width=True)

# Summary report
st.markdown('<h2 class="step-header">📊 Analysis Summary</h2>', unsafe_allow_html=True)

summary_html = f"""
<div class="metric-box">
<h4>🎯 Model Summary</h4>
<ul>
<li><strong>Dataset size:</strong> {len(clean_sizes)} samples</li>
<li><strong>Model type:</strong> Simple Linear Regression</li>
<li><strong>Performance:</strong> R² = {test_r2:.4f}, RMSE = {test_rmse:.2f} 10K CNY</li>
<li><strong>Learned equation:</strong> Price = {slope:.4f} × Area + {intercept:.4f}</li>
<li><strong>True equation:</strong> Price = {slope_param:.4f} × Area + {intercept_param:.4f}</li>
<li><strong>Parameter accuracy:</strong> Slope error = {abs(slope - slope_param):.4f}</li>
<li><strong>Business insight:</strong> Each 1 sqm increase leads to {slope:.2f} 10K CNY price increase</li>
<li><strong>Correlation:</strong> {correlation:.4f}</li>
</ul>
</div>
"""

st.markdown(summary_html, unsafe_allow_html=True)

# Download model parameters
if st.button("💾 Download Model Parameters"):
    model_params = f"""Linear Regression Model Parameters
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

User Settings:
- True Slope: {slope_param}
- True Intercept: {intercept_param}
- Noise Level: {noise_level}
- Number of Samples: {n_samples}
- Test Set Ratio: {test_size}

Learned Parameters:
- Learned Slope: {slope}
- Learned Intercept: {intercept}

Performance Metrics:
- Training R²: {train_r2}
- Test R²: {test_r2}
- Test RMSE: {test_rmse}
- Test MAE: {test_mae}

Model Equation:
Price = {slope:.6f} × Area + {intercept:.6f}
"""
    
    st.download_button(
        label="📄 Download as TXT",
        data=model_params,
        file_name=f"model_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🎓 Interactive Linear Regression Analysis using CRISP-DM Methodology</p>
<p>Built with Streamlit • Real-time Parameter Control • Educational Purpose</p>
</div>
""", unsafe_allow_html=True)