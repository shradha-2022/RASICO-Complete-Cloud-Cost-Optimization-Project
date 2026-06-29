import boto3
from datetime import datetime, timedelta

class AWSLiveCollector:
    def __init__(self, region="us-east-1"):
        self.region = region
        self.ec2 = boto3.client("ec2", region_name=region)
    
    def fetch_live_spot_prices(self, instance_types=None):
        params = {
            "StartTime": datetime.now() - timedelta(hours=1),
            "EndTime": datetime.now(),
            "ProductDescriptions": ["Linux/UNIX"]
        }
        if instance_types:
            params["InstanceTypes"] = instance_types
        response = self.ec2.describe_spot_price_history(**params)
        prices = []
        for price in response["SpotPriceHistory"]:
            prices.append({
                "timestamp": price["Timestamp"].isoformat(),
                "instance_type": price["InstanceType"],
                "spot_price": float(price["SpotPrice"]),
                "region": self.region
            })
        return prices
