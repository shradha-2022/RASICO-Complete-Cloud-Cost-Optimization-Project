"""
RASICO Phase 2 - Train XGBoost Models for Price & Risk Prediction
"""
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RASICO Phase 2: Training ML Models")
print("=" * 60)

# Create synthetic training data (replace with real AWS data later)
np.random.seed(42)

# Define instance types
INSTANCE_TYPES = ['t3.micro', 't3.small', 't3.medium', 'm5.large', 'm5.xlarge', 'c5.large', 'c5.xlarge']
REGIONS = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ap-northeast-1']

# Generate synthetic historical data
data = []
for _ in range(10000):
    region = np.random.choice(REGIONS)
    instance = np.random.choice(INSTANCE_TYPES)
    
    # Features
    hour = np.random.randint(0, 24)
    day_of_week = np.random.randint(0, 7)
    price_volatility = np.random.uniform(0.005, 0.05)
    price_to_ondemand_ratio = np.random.uniform(0.2, 0.8)
    interruption_freq = np.random.uniform(0.01, 0.4)
    
    # Encode features
    region_encoded = REGIONS.index(region)
    instance_family = 0 if 't3' in instance else 1 if 'm5' in instance else 2
    instance_size = int(instance.split('.')[-1].replace('micro', '0').replace('small', '1').replace('medium', '2').replace('large', '3').replace('xlarge', '4').replace('2xlarge', '5').replace('4xlarge', '6') if instance.split('.')[-1].isdigit() else 3)
    os_type = 0  # Linux
    
    # Base price by instance type
    base_prices = {'t3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
                   'm5.large': 0.096, 'm5.xlarge': 0.192, 'c5.large': 0.085, 'c5.xlarge': 0.170}
    on_demand = base_prices.get(instance, 0.05)
    
    # Target: next hour price (with realistic patterns)
    next_price = on_demand * price_to_ondemand_ratio
    # Add hourly pattern
    if hour in [9, 10, 11, 19, 20, 21]:
        next_price *= 1.15  # Peak hours
    if day_of_week >= 5:
        next_price *= 0.9  # Weekend discount
    # Add volatility
    next_price += np.random.normal(0, price_volatility * on_demand)
    next_price = max(0.001, min(next_price, on_demand * 0.95))
    
    # Target: interruption risk (binary)
    risk = 1 if (price_volatility > 0.03 and interruption_freq > 0.2 and hour in [9,10,11,19,20,21]) else 0
    
    data.append({
        'region_encoded': region_encoded,
        'instance_family': instance_family,
        'instance_size': instance_size,
        'os_type': os_type,
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'price_volatility': price_volatility,
        'price_to_ondemand_ratio': price_to_ondemand_ratio,
        'interruption_frequency_30d': interruption_freq,
        'next_hour_price': next_price,
        'high_interruption_risk': risk
    })

df = pd.DataFrame(data)

# Features for training
FEATURES = ['region_encoded', 'instance_family', 'instance_size', 'os_type',
            'hour_of_day', 'day_of_week', 'price_volatility',
            'price_to_ondemand_ratio', 'interruption_frequency_30d']

X = df[FEATURES]
y_price = df['next_hour_price']
y_risk = df['high_interruption_risk']

# Split data
X_train, X_test, y_price_train, y_price_test, y_risk_train, y_risk_test = train_test_split(
    X, y_price, y_risk, test_size=0.2, random_state=42
)

print(f"\n📊 Training Data: {len(X_train)} samples")
print(f"📊 Test Data: {len(X_test)} samples")

# Train Price Prediction Model (XGBoost Regressor)
print("\n🤖 Training Price Prediction Model...")
price_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
price_model.fit(X_train, y_price_train)

# Evaluate Price Model
y_price_pred = price_model.predict(X_test)
price_mae = mean_absolute_error(y_price_test, y_price_pred)
price_r2 = r2_score(y_price_test, y_price_pred)
print(f"   ✅ Price Model - MAE: ${price_mae:.5f}, R²: {price_r2:.4f}")

# Train Risk Prediction Model (XGBoost Classifier)
print("\n🤖 Training Interruption Risk Model...")
risk_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
risk_model.fit(X_train, y_risk_train)

# Evaluate Risk Model
y_risk_pred = risk_model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
risk_acc = accuracy_score(y_risk_test, y_risk_pred)
risk_f1 = f1_score(y_risk_test, y_risk_pred)
print(f"   ✅ Risk Model - Accuracy: {risk_acc:.4f}, F1: {risk_f1:.4f}")

# Save models
os.makedirs('models', exist_ok=True)
with open('models/price_model.pkl', 'wb') as f:
    pickle.dump(price_model, f)
with open('models/risk_model.pkl', 'wb') as f:
    pickle.dump(risk_model, f)

print("\n💾 Models saved to 'models/' folder")

print("\n" + "=" * 60)
print("✅ Phase 2 ML Models Training Complete!")
print("=" * 60)