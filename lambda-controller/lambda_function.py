import json
import boto3
import traceback

client = boto3.client('iot-data')

def lambda_handler(event, context):
    print('event: ', event)
    
    type = event['type']
    print('type: ', type)
    user_id = event['user_id'] # thingName
    print('user_id: ', user_id)
    thingName = user_id
    
    request_id = event['request_id']
    print('request_id: ', request_id)
    
    isAction = True
    if type == 'text':
        message = event['message']
        payload = json.dumps({
            "say": message, 
        })
    elif type == 'commend':
        commend = event['commend']
        print('commend: ', commend)

        payload = commend
    else:
        score = event['score']
        print('score: ', score)
    
        if score == '5':
            show = 'HAPPY'
            move = 'seq'
            seq = ["MOVE_FORWARD", "SIT", "MOVE_BACKWARD"]
        elif score == '4':
            show = 'NEUTRAL'
            move = 'seq'
            seq = ["TURN_LEFT", "SIT", "TURN_RIGHT"]
        elif score == '3':
            #show = 'NEUTRAL'
            #move = 'seq'
            #seq = ["LOOK_LEFT","LOOK_RIGHT", "LOOK_LEFT"]
            isAction = False
        elif score == '2':
            show = 'SAD'
            move = 'seq'
            seq = ["MOVE_BACKWARD", "SIT", "MOVE_FORWARD"]
        elif score == '1':
            show = 'ANGRY'
            move = 'seq'
            seq = ["LOOK_LEFT","LOOK_RIGHT", "LOOK_LEFT", "LOOK_RIGHT"]
        else:
            show = 'NEUTRAL'
            move = 'seq'
            seq = ["LOOK_LEFT","LOOK_RIGHT", "LOOK_LEFT", "LOOK_RIGHT"]
        
        payload = json.dumps({
            "show": show,  
            "move": move, 
            "seq": seq
        })
        
    topic = f"pupper/do/{thingName}"
    print('topic: ', topic)

    if isAction == True:    
        try:         
            response = client.publish(
                topic = topic,
                qos = 1,
                payload = payload
            )
            print('response: ', response)        
                
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")

    return {
        'statusCode': 200,
        'info': json.dumps({
            'result': 'success'
        })
    }
