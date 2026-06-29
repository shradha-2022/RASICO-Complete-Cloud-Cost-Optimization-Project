terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_sns_topic" "rasico_alerts" {
  name = "rasico-cost-alerts-${var.environment}"
}

resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.rasico_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_dashboard" "rasico" {
  dashboard_name = "RASICO-Monitoring-${var.environment}"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["RASICO", "SpotPrice"]
          ]
          period = 300
          stat = "Average"
          region = var.aws_region
          title = "Live Spot Prices"
          view = "timeSeries"
        }
      }
    ]
  })
}

output "sns_topic_arn" {
  value = aws_sns_topic.rasico_alerts.arn
}