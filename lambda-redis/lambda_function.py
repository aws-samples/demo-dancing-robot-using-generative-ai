import boto3
import os
import time
import re
import base64
import boto3
import uuid
import json
import redis
import traceback

# for Redis
redisAddress = os.environ.get('redisAddress')
redisPort = os.environ.get('redisPort')

try: 
    redis_client = redis.Redis(host=redisAddress, port=redisPort, db=0, charset="utf-8", decode_responses=True)    
except Exception:
    err_msg = traceback.format_exc()
    print('error message: ', err_msg)                    
    raise Exception ("Not able to request to LLM")

def push_game_event(state):
    list = {
        "AI-Dancing-Robot-000",
        "AI-Dancing-Robot-001",
        "AI-Dancing-Robot-002",
        "AI-Dancing-Robot-003",
        "AI-Dancing-Robot-004",
        "AI-Dancing-Robot-005",
        "AI-Dancing-Robot-006",
        "AI-Dancing-Robot-007",
        "AI-Dancing-Robot-008",
        "AI-Dancing-Robot-jinseony",
        "AI-Dancing-Robot-kyoungsu"        
    }
    
    for userId in list:    
        msg = {
            "type": "game",
            "userId": userId,
            "requestId": str(uuid.uuid4()),
            "query": "",
            "state": state
        }
        
        channel = f"{userId}"   
        try: 
            redis_client.publish(channel=channel, message=json.dumps(msg))
            print('successfully published: ', json.dumps(msg))
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")

def push_photo_event(userId, state):
    msg = {
        "type": "photo",
        "userId": userId,
        "requestId": str(uuid.uuid4()),
        "query": "",
        "state": state
    }
        
    channel = f"{userId}"   
    try: 
        redis_client.publish(channel=channel, message=json.dumps(msg))
        print('successfully published: ', json.dumps(msg))
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
   
def broadcast_message(userId, state, msg):
    msg = {
        "type": "broadcast",
        "userId": userId,
        "requestId": str(uuid.uuid4()),
        "query": msg,
        "state": state
    }
        
    channel = f"{userId}"   
    try: 
        redis_client.publish(channel=channel, message=json.dumps(msg))
        print('successfully published: ', json.dumps(msg))
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
       
requestId = str(uuid.uuid4())          
def lambda_handler(event, context):
    global requestId
    print('event: ', json.dumps(event))
    
    userId = event['userId']        
    # print('userId: ', userId)
    state = event['state']    
    
    if state == 'start' or state == 'end':  # game event
        push_game_event(state)
        
    elif state == 'start-photo' or state == 'end-photo':  # photo booth event
        push_photo_event(userId, state)
    
    elif state == 'broadcast':  
        broadcast_message(userId, state, event['message'])
        
    else: # user input
        if state == 'completed':
            requestId = str(uuid.uuid4())  
            
        if 'query' in event: 
            query = event['query']    
        else: 
            query = ""
            
        msg = {
            "type": "message",
            "userId": userId,
            "requestId": requestId,
            "query": query,
            "state": state
        }            
        
        channel = f"{userId}"   
        try: 
            redis_client.publish(channel=channel, message=json.dumps(msg))
            print('successfully published: ', json.dumps(msg))
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request")
            
        msg = "success"
        
    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps({ 
            "userId": userId,
            "state": state
        })
    }