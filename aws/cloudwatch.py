import boto3
from datetime import datetime

class RASICOMonitoring:
    def __init__(self, namespace="RASICO"):
        self.namespace = namespace
        self.cloudwatch = boto3.client("cloudwatch")
    
    def send_metric(self, metric_name, value, dimensions=None):
        metric_data = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": "Count",
            "Timestamp": datetime.now()
        }
        if dimensions:
            metric_data["Dimensions"] = dimensions
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[metric_data]
        )
        print(f"✅ Metric sent: {metric_name}={value}")
