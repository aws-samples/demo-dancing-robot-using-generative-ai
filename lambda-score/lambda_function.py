import os
import time
import boto3
import json
import re

import traceback
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"


def get_chat(region, model_id, max_output_token):
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=Config(
            retries={
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens": max_output_token,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    
    chat = ChatBedrock(   
        model_id=model_id,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )  

    return chat

def get_lambda_client(region):
    # bedrock
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

def get_prompt():
    system = ("""
너는 사람이 말을 했을 때 강아지가 받아들이는 감정을 측정하는 AI 란다.

아래 <text> 태그 안의 말을 했을 때 <character> 안에 있는 성격을 가진 강아지가 어떻게 받아들이는지 점수를 계산해줘
계산한 점수는 <score> 태그 안에 넣어서 출력해줘. 점수는 5점부터 1점까지야.
주인의 말을 많이 좋아할수록 5점이고 조금 좋아할 수록 1점이야.

점수에 대한 이유는 5토큰 이내로 줄여서 <description> 태그 안에 넣어서 출력해줘
    """
              )

    human = """
<text>
{text}
</text>

<character>
{character}
</character>
"""

    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])

def get_character(mbti):
    # ESTJ 로 점수 통일
    mbti = "ESTJ"

    mbti_character = {
        "ISTP": """
만사 귀찮아함
하기싫은건 죽어도 안함
과묵하지만 호기심은 강함
내향적이며 논리적임
집에 있는 것을 좋아함
""",
        "ESFP": """
분위기메이커이며 목소리 큰 편임
친화력이 좋고 사교성 적응력 뛰어남
다른 사람 자존감 높여주는 말을 잘함
이야기할때 부연 설명이 많음
남의 기분을 잘 공감함
""",
        "INFJ": """
생각이 너무 많음
주인에게 매우 잘해줌
계획적인거 좋아 함
단 둘이 노는거 좋아함
관심 받고 싶은데 나서는건 싫어함
""",
        "ESTJ": """
주인에게 충성함
전투적인 성격이 있음
부지런한 성격
게으른거를 싫어함
고집이 강함
"""
    }
    return mbti_character[mbti]


def extract_text_from_tags(text, tag):
    try:
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)[0]
    except Exception:
        return ''
    return matches


def extract_sentiment(chat, text, mbti):

    prompt = get_prompt()
    character = get_character(mbti)

    chain = prompt | chat
    try:
        result = chain.invoke(
            {
                "text": text,
                "character":  character
            }
        )
        msg = result.content
        # print('result of sentiment extraction: ', msg)

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
        raise Exception("Not able to request to LLM")

    score = extract_text_from_tags(msg, "score")
    description = extract_text_from_tags(msg, "description")

    return {
        'score': score,
        'description': description
    }

def lambda_handler(event, context):
    try:
        selected_LLM = 0
        profile_of_LLMs = json.loads(os.environ.get('profile_of_LLMs'))
        profile = profile_of_LLMs[selected_LLM]
    except Exception:
        profile = {
            'bedrock_region': 'us-west-2',
            'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'maxOutputTokens': '1024'
        }

    bedrock_region = profile['bedrock_region']
    model_id = profile['model_id']
    max_output_token = int(profile['maxOutputTokens'])

    print(f'bedrock_region: {bedrock_region}, modelId: {model_id}, max_output_token: {max_output_token}')
    print('event: ', event)

    body = event.get("body", "")
    jsonBody = json.loads(body)

    user_id = jsonBody["userId"]
    request_id = jsonBody["requestId"]

    text = jsonBody["text"]
    mbti = jsonBody["mbti"]

    start_time_for_greeting = time.time()

    # creating greeting message
    chat = get_chat(region=bedrock_region, model_id=model_id, max_output_token=max_output_token)
    result = extract_sentiment(chat=chat, text=text, mbti=mbti)
    print('result: ', result)

    # ToDo
    # 스코어 보드 호출
    function_name = "lambda-score-update-for-demo-dansing-robot"
    lambda_region = 'ap-northeast-2'
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {
            "userId": user_id,
            "score": result['score'],
            "type": "MENT",
            "body": text
        }
        print("payload: ", payload)
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise

    end_time_for_greeting = time.time()
    time_for_greeting = end_time_for_greeting - start_time_for_greeting

    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps({
            "userId": user_id,
            "requestId": request_id,
            "result": result,
            "time_taken": str(time_for_greeting)
        }, ensure_ascii=False)
    }
