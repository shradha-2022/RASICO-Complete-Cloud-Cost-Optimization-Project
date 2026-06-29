# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from datetime import datetime
# import redis
# import os
# import sys
# sys.path.append(".")
# from aws.live_collector import AWSLiveCollector

# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def index():
#     return jsonify({"message": "RASICO v2.0 - Risk-Aware Spot Instance Optimizer", "status": "running"})

# @app.route("/api/health")
# def health():
#     return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# @app.route("/api/spot-prices")
# def get_prices():
#     region = request.args.get("region", "us-east-1")
#     collector = AWSLiveCollector(region=region)
#     prices = collector.fetch_live_spot_prices()
#     return jsonify({
#         "region": region,
#         "timestamp": datetime.now().isoformat(),
#         "count": len(prices),
#         "prices": prices[:10]
#     })

# @app.route("/api/regions")
# def get_regions():
#     regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]
#     return jsonify({"regions": regions})

# if __name__ == "__main__":
#     print("🚀 RASICO API Starting...")
#     app.run(host="0.0.0.0", port=5000, debug=True)




#  updated code for phase 2 ml
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import redis
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws.live_collector import AWSLiveCollector

app = Flask(__name__)
CORS(app)

# Load ML Models
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
price_model = None
risk_model = None

try:
    with open(os.path.join(MODELS_PATH, 'price_model.pkl'), 'rb') as f:
        price_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'risk_model.pkl'), 'rb') as f:
        risk_model = pickle.load(f)
    print("✅ ML Models loaded successfully")
except Exception as e:
    print(f"⚠️ Models not found: {e}")
    print("   Run 'python train_models.py' first")

# Instance specifications for recommendations
INSTANCE_SPECS = {
    't3.micro': {'vcpu': 2, 'memory': 1, 'family': 't3', 'on_demand': 0.0104, 'description': 'Burstable General Purpose'},
    't3.small': {'vcpu': 2, 'memory': 2, 'family': 't3', 'on_demand': 0.0208, 'description': 'Burstable General Purpose'},
    't3.medium': {'vcpu': 2, 'memory': 4, 'family': 't3', 'on_demand': 0.0416, 'description': 'Burstable General Purpose'},
    'm5.large': {'vcpu': 2, 'memory': 8, 'family': 'm5', 'on_demand': 0.0960, 'description': 'General Purpose'},
    'm5.xlarge': {'vcpu': 4, 'memory': 16, 'family': 'm5', 'on_demand': 0.1920, 'description': 'General Purpose'},
    'c5.large': {'vcpu': 2, 'memory': 4, 'family': 'c5', 'on_demand': 0.0850, 'description': 'Compute Optimized'},
    'c5.xlarge': {'vcpu': 4, 'memory': 8, 'family': 'c5', 'on_demand': 0.1700, 'description': 'Compute Optimized'}
}

REGIONS = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ap-northeast-1']
REGION_MAP = {r: i for i, r in enumerate(REGIONS)}
FAMILY_MAP = {'t3': 0, 'm5': 1, 'c5': 2}

def predict_price_and_risk(region, instance_type, hour=None, day_of_week=None):
    """Use ML models to predict price and risk"""
    if hour is None:
        hour = datetime.now().hour
    if day_of_week is None:
        day_of_week = datetime.now().weekday()
    
    instance = INSTANCE_SPECS[instance_type]
    
    # Prepare features
    features = np.array([[
        REGION_MAP.get(region, 0),
        FAMILY_MAP.get(instance['family'], 0),
        int(instance_type.split('.')[-1].replace('micro', '0').replace('small', '1').replace('medium', '2')
            .replace('large', '3').replace('xlarge', '4').replace('2xlarge', '5').replace('4xlarge', '6')) if instance_type.split('.')[-1].isdigit() else 3,
        0,  # os_type (Linux)
        hour,
        day_of_week,
        0.02,  # default volatility
        0.35,  # default price ratio
        0.1    # default interruption freq
    ]])
    
    # Predict
    if price_model:
        predicted_price = float(price_model.predict(features)[0])
        # Ensure price is reasonable
        predicted_price = max(0.001, min(predicted_price, instance['on_demand'] * 0.95))
    else:
        # Fallback: 70% of on-demand
        predicted_price = instance['on_demand'] * 0.7
    
    if risk_model:
        risk_prob = float(risk_model.predict_proba(features)[0][1])
    else:
        risk_prob = 0.15
    
    # Adjust risk based on region and hour
    if region in ['ap-southeast-1', 'ap-northeast-1']:
        risk_prob *= 1.2
    if hour in [9, 10, 11, 19, 20, 21]:
        risk_prob *= 1.3
    
    risk_prob = min(0.95, risk_prob)
    
    return predicted_price, risk_prob

def calculate_rasico_score(price, on_demand, risk, alpha=0.5):
    """RASICO Score = α * (price/on_demand) + (1-α) * risk"""
    price_norm = min(price / on_demand, 1.0)
    return round(alpha * price_norm + (1 - alpha) * risk, 6)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/spot-prices')
def get_prices():
    region = request.args.get('region', 'us-east-1')
    try:
        collector = AWSLiveCollector(region=region)
        prices = collector.fetch_live_spot_prices()
        return jsonify({
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "count": len(prices),
            "prices": prices[:20]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/regions')
def get_regions():
    return jsonify({"regions": REGIONS})

@app.route('/api/instances')
def get_instances():
    return jsonify({"instances": list(INSTANCE_SPECS.keys())})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get ML-powered recommendations"""
    data = request.json or {}
    alpha = float(data.get('alpha', 0.5))
    top_n = int(data.get('top_n', 5))
    target_family = data.get('target_family', None)
    workload = data.get('workload', 'general')
    
    recommendations = []
    now = datetime.now()
    
    for region in REGIONS:
        for instance_type, specs in INSTANCE_SPECS.items():
            if target_family and specs['family'] != target_family:
                continue
            
            # Get ML predictions
            predicted_price, risk_prob = predict_price_and_risk(region, instance_type, now.hour, now.weekday())
            on_demand = specs['on_demand']
            
            # Calculate metrics
            savings_pct = round((on_demand - predicted_price) / on_demand * 100, 1)
            rasico_score = calculate_rasico_score(predicted_price, on_demand, risk_prob, alpha)
            
            # Risk label
            if risk_prob < 0.25:
                risk_label = "LOW"
                risk_color = "green"
            elif risk_prob < 0.5:
                risk_label = "MEDIUM"
                risk_color = "orange"
            else:
                risk_label = "HIGH"
                risk_color = "red"
            
            # Monthly projections
            monthly_savings = round((on_demand - predicted_price) * 730, 2)
            
            recommendations.append({
                'region': region,
                'instance_type': instance_type,
                'vcpu': specs['vcpu'],
                'memory_gb': specs['memory'],
                'description': specs['description'],
                'on_demand_price': round(on_demand, 5),
                'predicted_price': round(predicted_price, 5),
                'savings_percentage': savings_pct,
                'monthly_savings': monthly_savings,
                'risk_probability': round(risk_prob, 3),
                'risk_label': risk_label,
                'risk_color': risk_color,
                'rasico_score': rasico_score,
                'recommended_for': 'Batch jobs' if savings_pct > 60 else 'Web servers' if risk_prob < 0.3 else 'Dev/Test'
            })
    
    # Sort by RASICO score (lower is better)
    recommendations.sort(key=lambda x: x['rasico_score'])
    
    return jsonify({
        'status': 'success',
        'query': {
            'alpha': alpha,
            'workload': workload,
            'timestamp': now.isoformat()
        },
        'total_candidates': len(recommendations),
        'recommendations': recommendations[:top_n],
        'best': recommendations[0] if recommendations else None,
        'models_loaded': price_model is not None and risk_model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict price and risk for specific instance"""
    data = request.json
    region = data.get('region', 'us-east-1')
    instance_type = data.get('instance_type', 't3.medium')
    hour = data.get('hour', datetime.now().hour)
    day = data.get('day', datetime.now().weekday())
    
    predicted_price, risk_prob = predict_price_and_risk(region, instance_type, hour, day)
    on_demand = INSTANCE_SPECS[instance_type]['on_demand']
    
    return jsonify({
        'region': region,
        'instance_type': instance_type,
        'current_on_demand': on_demand,
        'predicted_spot_price': round(predicted_price, 5),
        'savings_percentage': round((on_demand - predicted_price) / on_demand * 100, 1),
        'risk_probability': round(risk_prob, 3),
        'risk_label': 'LOW' if risk_prob < 0.25 else 'MEDIUM' if risk_prob < 0.5 else 'HIGH',
        'rasico_score': calculate_rasico_score(predicted_price, on_demand, risk_prob)
    })

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare two instances side by side"""
    data = request.json
    instance1 = data.get('instance1', 't3.medium')
    instance2 = data.get('instance2', 'm5.large')
    region = data.get('region', 'us-east-1')
    
    def get_instance_data(instance_type):
        price, risk = predict_price_and_risk(region, instance_type)
        on_demand = INSTANCE_SPECS[instance_type]['on_demand']
        return {
            'instance_type': instance_type,
            'predicted_price': round(price, 5),
            'on_demand_price': on_demand,
            'savings_percentage': round((on_demand - price) / on_demand * 100, 1),
            'risk_probability': round(risk, 3),
            'rasico_score': calculate_rasico_score(price, on_demand, risk)
        }
    
    return jsonify({
        'region': region,
        'instance1': get_instance_data(instance1),
        'instance2': get_instance_data(instance2)
    })
@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export recommendations to CSV file"""
    import csv
    import io
    from flask import Response
    
    data = request.json or {}
    alpha = float(data.get('alpha', 0.5))
    target_family = data.get('target_family', None)
    
    # Generate recommendations
    recommendations = []
    now = datetime.now()
    
    for region in REGIONS:
        for instance_type, specs in INSTANCE_SPECS.items():
            if target_family and specs['family'] != target_family:
                continue
            
            predicted_price, risk_prob = predict_price_and_risk(region, instance_type, now.hour, now.weekday())
            on_demand = specs['on_demand']
            savings_pct = round((on_demand - predicted_price) / on_demand * 100, 1)
            rasico_score = calculate_rasico_score(predicted_price, on_demand, risk_prob, alpha)
            
            if risk_prob < 0.25:
                risk_label = "LOW"
            elif risk_prob < 0.5:
                risk_label = "MEDIUM"
            else:
                risk_label = "HIGH"
            
            monthly_savings = round((on_demand - predicted_price) * 730, 2)
            
            recommendations.append({
                'Rank': len(recommendations) + 1,
                'Instance Type': instance_type,
                'vCPU': specs['vcpu'],
                'Memory (GB)': specs['memory'],
                'Region': region,
                'On-Demand Price ($/hr)': on_demand,
                'Predicted Spot Price ($/hr)': round(predicted_price, 5),
                'Savings (%)': savings_pct,
                'Monthly Savings ($)': monthly_savings,
                'Risk Level': risk_label,
                'Risk Probability (%)': round(risk_prob * 100, 1),
                'RASICO Score': rasico_score,
                'Best For': 'Batch jobs' if savings_pct > 60 else 'Web servers' if risk_prob < 0.3 else 'Dev/Test'
            })
    
    # Sort by RASICO score
    recommendations.sort(key=lambda x: x['RASICO Score'])
    
    # Create CSV
    output = io.StringIO()
    if recommendations:
        writer = csv.DictWriter(output, fieldnames=recommendations[0].keys())
        writer.writeheader()
        writer.writerows(recommendations)
    
    # Return CSV file
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=rasico_recommendations.csv'}
    )

# launch instance buuton functionaltiy
@app.route('/api/launch', methods=['POST'])
def launch_instance():
    """Launch a real EC2 spot instance on AWS"""
    import boto3
    
    data = request.json
    instance_type = data.get('instance_type', 't3.medium')
    region = data.get('region', 'us-east-1')
    spot_price = data.get('spot_price', 0.02)
    
    try:
        ec2 = boto3.client('ec2', region_name=region)
        
        # Fixed AMI IDs for each region (verified working)
        region_amis = {
            'us-east-1': 'ami-02b9a589195146a8f',
            'us-west-2': 'ami-0c0f2fbfe22ff7e5d',
            'eu-west-1': 'ami-0008e653e8cd339b7',
            'ap-southeast-1': 'ami-0a8298b7d9dfb7c0c',
            'ap-northeast-1': 'ami-0b3c625f8f2c18c1b'
        }
        
        ami_id = region_amis.get(region)
        
        if not ami_id:
            return jsonify({
                'success': False,
                'error': f'No AMI configured for region {region}. Try us-east-1.'
            }), 400
        
        print(f"Using AMI: {ami_id} for region {region}")
        
        # Request spot instance
        response = ec2.request_spot_instances(
            SpotPrice=str(spot_price),
            InstanceCount=1,
            LaunchSpecification={
                'ImageId': ami_id,
                'InstanceType': instance_type,
                'SecurityGroupIds': []
            }
        )
        
        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        
        return jsonify({
            'success': True,
            'message': f'✅ Spot instance launching!',
            'request_id': request_id,
            'instance_type': instance_type,
            'region': region,
            'spot_price': spot_price
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 RASICO Phase 2 - ML-Powered Recommendation Engine")
    print("=" * 60)
    print("\n✅ ML Models: " + ("LOADED" if price_model else "NOT LOADED (run train_models.py)"))
    print("\n📌 Available endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/spot-prices?region=us-east-1")
    print("  POST /api/recommend")
    print("  POST /api/predict")
    print("  POST /api/compare")
    print("\n📍 Dashboard: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)