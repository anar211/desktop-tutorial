# Install required packages: pip install numpy pandas scikit-learn matplotlib seaborn shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic customer churn dataset
# Features: tenure (months), monthly_charges ($), contract_type (0: month-to-month, 1: one-year, 2: two-year),
#          support_calls (#), data_usage (GB), payment_delay (days)
n_samples = 1000
data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'contract_type': np.random.choice([0, 1, 2], n_samples),
    'support_calls': np.random.randint(0, 10, n_samples),
    'data_usage': np.random.uniform(1, 100, n_samples),
    'payment_delay': np.random.randint(0, 30, n_samples)
}
df = pd.DataFrame(data)

# Generate churn target (1: churn, 0: no churn) based on a simple rule with noise
df['churn'] = ((df['tenure'] < 12) & (df['monthly_charges'] > 80) | (df['support_calls'] > 5)).astype(int)
df['churn'] = df['churn'] * np.random.choice([0, 1], n_samples, p=[0.2, 0.8])  # Add noise
df['churn'] = df['churn'].clip(0, 1)

# 2. Preprocess and split data
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# 4. Calculate SHAP values as baseline technical metric
# Use SHAP's TreeExplainer for Random Forest to compute feature contributions
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
# For binary classification, use SHAP values for class 1 (churn)
mean_abs_shap = np.mean(np.abs(shap_values[1]), axis=0).mean() * 10  # Normalize to 0-10 scale

# 5. Define Robust Simplicity Score (RSS)
# RSS = (Feature Clarity / Explanation Complexity) * Normalization Factor
# Feature Clarity: Inverse of entropy of feature importances (higher clarity if fewer features dominate)
# Explanation Complexity: Number of features with non-zero importance
def calculate_rss(model, X):
    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(X.shape[1]) / X.shape[1]
    feature_clarity = 1 / (entropy(feature_importances) + 1e-10)  # Avoid division by zero
    explanation_complexity = np.sum(feature_importances > 0)  # Count non-zero features
    rss = (feature_clarity / (explanation_complexity + 1)) * 10  # Normalize to 0-10 scale
    return rss

rss_rf = calculate_rss(rf_model, X_train)

# 6. Define Trust Scope Index (TSI)
# TSI = w1 * Domain Alignment + w2 * Stakeholder Trust
# Domain Alignment: Cosine similarity between feature importances and domain knowledge weights
# Stakeholder Trust: Simulated user feedback score (0-1) based on explanation interpretability
def calculate_tsi(model, X, domain_weights, stakeholder_feedback=0.8):
    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(X.shape[1]) / X.shape[1]
    # Normalize feature importances and domain weights
    feature_importances = feature_importances / (np.sum(feature_importances) + 1e-10)
    domain_weights = np.array(domain_weights) / (np.sum(domain_weights) + 1e-10)
    # Cosine similarity for domain alignment
    domain_alignment = np.dot(feature_importances, domain_weights) / (
        np.linalg.norm(feature_importances) * np.linalg.norm(domain_weights) + 1e-10)
    # Weighted combination (equal weights for simplicity)
    tsi = 0.5 * domain_alignment + 0.5 * stakeholder_feedback
    return tsi * 10  # Normalize to 0-10 scale

# Domain knowledge: tenure and contract_type are most important for churn
domain_weights = [0.4, 0.2, 0.3, 0.05, 0.03, 0.02]  # Sum to 1
tsi_rf = calculate_tsi(rf_model, X_train, domain_weights, stakeholder_feedback=0.8)

# 7. Visualization of explainability metrics
plt.figure(figsize=(10, 6))
metrics = ['Mean Abs SHAP Value', 'Robust Simplicity Score (RSS)', 'Trust Scope Index (TSI)']
values = [mean_abs_shap, rss_rf, tsi_rf]
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title('Explainability Metrics for Customer Churn Model')
plt.ylabel('Metric Value (Normalized)')
plt.xticks(rotation=15)
for i, v in enumerate(values):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()

# Save and show the plot
plt.savefig('explainability_metrics.png')
plt.show()

# Print model performance and metrics for reference
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Mean Absolute SHAP Value: {mean_abs_shap:.2f}")
print(f"Robust Simplicity Score (RSS): {rss_rf:.2f}")
print(f"Trust Scope Index (TSI): {tsi_rf:.2f}")