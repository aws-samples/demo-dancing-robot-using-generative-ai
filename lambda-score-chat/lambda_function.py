import json
import boto3
import os
from botocore.exceptions import ClientError

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

def send_dashboard(userId, text, type):
    # 스코어 보드 호출
    function_name = "lambda-score-update-for-demo-dansing-robot"
    
    lambda_region = 'ap-northeast-2'
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {
            "userId": userId,
            "text": text,
            "type": type
        }
        print("Payload: ", payload)
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise

def lambda_handler(event, context):
    print('event: ', event)
    
    userId = event["user_id"]
    requestId = event["request_id"]
    type = event["type"]
    text = event["text"]
    
    # send message to the dashboard
    send_dashboard(userId, text, type)
    
    # To-do: Push the text for the last message   
    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps({        
            "userId": userId,
            "requestId": requestId,
            "type": type
        })
    }
