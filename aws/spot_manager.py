import boto3

class SpotManager:
    def __init__(self, region="us-east-1"):
        self.ec2 = boto3.client("ec2", region_name=region)
    
    def request_spot_instance(self, instance_type, ami_id, max_price):
        response = self.ec2.request_spot_instances(
            SpotPrice=str(max_price),
            InstanceCount=1,
            LaunchSpecification={
                "InstanceType": instance_type,
                "ImageId": ami_id
            }
        )
        return response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
