import json
import boto3
import os
from botocore.exceptions import ClientError

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )


def get_score(text):
    # "heart", "X", "O", "1 thumb-up", "1 thumb-down", "2 thumb-up", "2 thumb-down", "1 victory", "2 victory"

    if text == "heart":
        score = 5
    elif text == "2 thumb-up":
        score = 5
    elif text == "2 victory":
        score = 5
    elif text == "1 thumb-up":
        score = 4
    elif text == "1 victory":
        score = 4
    elif text == "2 thumb-down":
        score = 1
    elif text == "1 thumb-down":
        score = 2
    else: # "X", "O"
        score = 1

    return score


def send_dashboard(userId, score, text, type):
    # 스코어 보드 호출
    function_name = "lambda-score-update-for-demo-dansing-robot"
    
    lambda_region = 'ap-northeast-2'
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {
            'userId': userId,
            'score': score,
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
    
    userId = event["userId"]
    requestId = event["requestId"]
    type = event["type"]
    text = event["text"]
    
    # Get Score
    if type == 'gesture':
        score = get_score(text)
    print('score: ', score)

    # send the score to the dashboard
    send_dashboard(userId, score, text, type)
    
    # To-do: Push the text for the last message   
    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps({        
            "userId": userId,
            "requestId": requestId,
            "score": score
        })
    }
