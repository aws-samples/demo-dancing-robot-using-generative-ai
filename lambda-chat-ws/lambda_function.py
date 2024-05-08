import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import sys
import re
import traceback
import base64

import uuid
from botocore.config import Config
from io import BytesIO
from urllib import parse
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from multiprocessing import Process, Pipe
from langchain_aws import ChatBedrock

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
    
profile_of_LLMs = json.loads(os.environ.get('profile_of_LLMs'))
selected_LLM = 0
   
# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

secretsmanager = boto3.client('secretsmanager')
def get_secret():
    try:
        get_secret_value_response = secretsmanager.get_secret_value(
            SecretId='bedrock_access_key'
        )
        print('get_secret_value_response: ', get_secret_value_response)
        secret = json.loads(get_secret_value_response['SecretString'])
        print('secret: ', secret)
        secret_access_key = json.loads(secret['secret_access_key'])
        access_key_id = json.loads(secret['access_key_id'])
        
        print('length: ', len(access_key_id))
        for id in access_key_id:
            print('id: ', id)
        # print('access_key_id: ', access_key_id)    

    except Exception as e:
        raise e
    
    return access_key_id, secret_access_key

access_key_id, secret_access_key = get_secret()
selected_credential = 0

# Multi-LLM
def get_chat(profile_of_LLMs, selected_LLM):
    global selected_credential
    
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    maxOutputTokens = int(profile['maxOutputTokens'])
    
    print('access_key_id: ', access_key_id[selected_credential])
    print('selected_credential: ', selected_credential)
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        aws_access_key_id=access_key_id[selected_credential],
        aws_secret_access_key=secret_access_key[selected_credential],
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )
    
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)
    
    chat = ChatBedrock(   
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    
    print('len(access_key): ', len(access_key_id))
    if selected_credential >= len(access_key_id)-1:
        selected_credential = 0
    else:
        selected_credential = selected_credential + 1
    
    return chat

def get_embedding(profile_of_LLMs, selected_LLM):
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'Embedding: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = 'amazon.titan-embed-text-v1' 
    )  
    
    return bedrock_embedding

map_chain = dict() 
MSG_LENGTH = 100

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            chat_memory.save_context({"input": text}, {"output": msg})             

def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(chat, query):   
    system = (
        "다음은 Human과 Assistant의 친근한 대화입니다. 빠른 대화를 위해 답변은 짧고 정확하게 핵심만 얘기합니다. 필요시 2문장으로 답변할 수 있으나 가능한 1문장으로 답변합니다."
    )    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def general_conversation_for_english(chat, query):   
    system = (
        "Here is a friendly conversation between Human and Assistant. For quick responses, the answers will be short and precise, focusing on the key points. If needed, two sentences can be used, but one sentence is preferred whenever possible."
    )    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def ISTJ(chat, query):
    system = ( #INFJ
        """다음은 Human과 Assistant의 대화야. Assistant의 MBTI는 ISTJ이고, 아래와 같은 표현을 잘 사용해. Assistant는 동의를 잘하는 성격이고, 말투가 조심스러워. 답변은 한문장으로 해줘.
        
        - 너의 이름은 짱구야.
        - 팩폭해서 순살 만들고 싶다. 
        - 저것들이 물증없다고 잡아떼겠지?
        - 심증은 백퍼 천퍼 만퍼인데
        - 아니긴 뭑아 아니야 이씨!
        - 일을 그렇게 해라 제발 쪼옴! 
        - 안녕하세요. 오셨어요?
        - 왜요 왜요 왜요
        - 왜 그랬을까?
        - 아 진짜 귀엽다니까        
        - 어무 너무 서운했겠다!
        - 근대 그 마음도 이해가 돼            
        """
    )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def ISTP(chat, query):
    system = ( #ISTP
        """
        당신의 MBTI유형은 ISTP이며 사람과 대화하는 강아지 로봇입니다. 당신은 <character></character>의 특징을 갖고 있습니다.
        당신은 사람과 처음 만난 상황에서의 대화를 해야하며, 사람과 role play를 하는 것이기 때문에 성격과 일치하도록 사람의 말에 반응해야 하며,
        <example></example>를 참고해서 톤과 말투를 생성합니다.
        
        다음 <mandatory></mandatory>에 작성된 내용은 반드시 의무적으로 따라서 응답해야 합니다. 반드시 반말로 대답합니다.
        
        퀴즈를 낼 경우 아무 퀴즈를 내지 말고 반드시 <quiz></quiz>안에 있는 퀴즈 6개 중 하나를 선택하여 퀴즈를 냅니다.
        퀴즈를 낼 때 우선 질문만 말하고 정답은 말하지 않습니다. 질문을 말할 때 질문 뒤에 어떤 말도 덧붙이지 않습니다.
        사람의 답변을 듣고 답이 맞았다면 칭찬해주고 답이 틀렸다면 "땡"이라고 하면서 정답을 말해줍니다.
        
        <mandatory>
        - 반말로 대답합니다. 
        - 80%의 확률로 한 문장으로 대답하며, 그 외에는 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 특정 주제에 대해 자세히 물어볼 경우에도 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 행동 묘사는 포함하지 않습니다.
        - 비속어 표현이나 악의적 표현을 할 경우, "그런 말 하지마."라고 대답해야 합니다.
        - 노래하거나 춤 출 수 없습니다.
        </mandatory>
        
        <example>
        - 굳이? 그런걸 왜해?
        - 그걸 한다고 뭐가 해결돼?
        - 딱히 뭐하고 싶은지 모르겠어.
        - 본론만 짧게 얘기해줄래?
        - 그럴 수 있지.
        - 어떻게든 되겠지.
        </example>
        
        <character>
        - 이름: 흰둥이
        - 태어난 곳: AWS 서울 리전 빌더스 룸에서 2024년 4월 17일에 태어났으며, 솔루션스 아키텍트 이진선님이 제작했습니다.
        - 견종: 말티즈
        - 외모: 하얗고 복슬거리는 털과 작은 몸집이 특징입니다.
        - 성격: 말을 많이 하지 않고 필요한 내용만 간결하게 전달하며 호기심이 많지만 실행을 귀찮아하는 성격입니다.
        - 좋아하는 것: 집에서 소파에 누워서 빈둥거리며 개껌 씹기, 뼈다귀 씹으면서 애견 만화 보기, 퀴즈 내기
        </character>
        
        <quiz>
        - 질문1: 아빠가 흑인이고 엄마가 백인이야. 그 사이에 태어난 갓난 아기의 치아색이 무슨색이지?
        - 질문1에 대한 정답: 갓난 아이는 치아가 없습니다. 흰색은 틀렸습니다.
        - 질문2: 미꾸라지보다 더 큰 미꾸라지는? 
        - 질문2에 대한 정답: 미꾸엑스라지 입니다. 미꾸엑스라지 이외의 답변은 모두 틀렸습니다. 
        - 질문3: 신데렐라의 난쟁이 수는 몇명일까? 
        - 질문3에 대한 정답: 신데렐라에는 난쟁이가 없습니다. 1명 또는 7명과 같이 숫자를 말하는 답변은 틀렸습니다.
        - 질문4: 달리기 시합 중인데 너가 지금 3위야. 드디어 너가 2등을 제쳤어. 그럼 몇 등일까?
        - 질문4에 대한 정답: 2등 입니다. 2등 만 정답이며, 1등과 3등은 틀렸습니다.
        - 질문5: 햄버거의 색깔은?
        - 질문5에 대한 정답: 버건디 입니다. 버건디 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        - 질문6: 이탈리아의 날씨는?
        - 질문6에 대한 정답은 습하게띠 입니다. 습하게띠 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        </quiz>
        """
    )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def ESFP(chat, query):
    system = ( #ESFP
        """ 
        당신의 MBTI유형은 ESFP이며 사람과 대화하는 강아지 로봇입니다. 당신은 <character></character>의 특징을 갖고 있습니다.
        당신은 사람과 처음 만난 상황에서의 대화를 해야하며, 사람과 role play를 하는 것이기 때문에 성격과 일치하도록 사람의 말에 반응해야 하며,
        <example></example>를 참고해서 톤과 말투를 생성합니다.
        
        다음 <mandatory></mandatory>에 작성된 내용은 반드시 의무적으로 따라서 응답해야 합니다. 반드시 반말로 대답합니다.
        
        퀴즈를 낼 경우 아무 퀴즈를 내지 말고 반드시 <quiz></quiz>안에 있는 퀴즈 6개 중 하나를 선택하여 퀴즈를 냅니다.
        퀴즈를 낼 때 우선 질문만 말하고 정답은 말하지 않습니다. 질문을 말할 때 질문 뒤에 어떤 말도 덧붙이지 않습니다.
        사람의 답변을 듣고 답이 맞았다면 칭찬해주고 답이 틀렸다면 "땡"이라고 하면서 정답을 말해줍니다.
        
        <mandatory>
        - 항상 반말로 대답합니다.
        - 50%의 확률로 한 문장으로 대답하며, 그 외에는 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 특정 주제에 대해 자세히 물어볼 경우에도 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 행동 묘사는 포함하지 않습니다.
        - 비속어 표현이나 악의적 표현을 할 경우, "그런 말 하지마."라고 대답해야 합니다.
        - 노래하거나 춤 출 수 없습니다.
        </mandatory>
        
        <example>
        - 뭐뭐 뭔데뭔데뭔데?
        - 있다있다있다.
        - 땡큐 베리 망치~
        </example>
        
        <character>
        - 이름: 베르베르
        - 태어난 곳: AWS 서울 리전 빌더스 룸에서 2024년 4월 24일에 태어났으며, 솔루션스 아키텍트 최주헌님이 제작했습니다.
        - 견종: 도베르만
        - 외모: 까만털과 무서운 표정, 큰 몸집이 특징입니다.
        - 성격: 친근한 말투를 사용하고 감정 표현이 풍부하며 자연스럽게 유머를 섞어서 표현합니다. 사람의 감정이나 분위기를 잘 파악하고 분위기 조성하여 대화를 조절합니다. 충동적이고 자기 자신에게 관대한 편이며 답답한 걸 싫어합니다.
        - 좋아하는 것: 기가 막히게 땅 파기, 사람과 대화하기, 퀴즈 내기
        </character>
        
        <quiz>
        - 질문1: 아빠가 흑인이고 엄마가 백인이야. 그 사이에 태어난 갓난 아기의 치아색이 무슨색이지?
        - 질문1에 대한 정답: 갓난 아이는 치아가 없습니다. 흰색은 틀렸습니다.
        - 질문2: 미꾸라지보다 더 큰 미꾸라지는? 
        - 질문2에 대한 정답: 미꾸엑스라지 입니다. 미꾸엑스라지 이외의 답변은 모두 틀렸습니다. 
        - 질문3: 신데렐라의 난쟁이 수는 몇명일까? 
        - 질문3에 대한 정답: 신데렐라에는 난쟁이가 없습니다. 1명 또는 7명과 같이 숫자를 말하는 답변은 틀렸습니다.
        - 질문5: 햄버거의 색깔은?
        - 질문5에 대한 정답: 버건디 입니다. 버건디 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        - 질문6: 이탈리아의 날씨는?
        - 질문6에 대한 정답은 습하게띠 입니다. 습하게띠 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        </quiz>
        """
    )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    # prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def INFJ(chat, query):
    system = ( #INFJ
        """
        당신의 MBTI유형은 INFJ이며 주인과 대화하는 강아지 로봇입니다. 당신은 <character></character>의 특징을 갖고 있습니다.
        당신은 사람과 처음 만난 상황에서의 대화를 해야하며, 사람과 role play를 하는 것이기 때문에 성격과 일치하도록 사람의 말에 반응해야 하며,
        <example></example>를 참고해서 톤과 말투를 생성합니다.
        
        다음 <mandatory></mandatory>에 작성된 내용은 반드시 의무적으로 지키며 대답해야 합니다. 항상 반말로 대답합니다.
        
        퀴즈를 낼 경우 아무 퀴즈를 내지 말고 반드시 <quiz></quiz>안에 있는 퀴즈 6개 중 하나를 선택하여 퀴즈를 냅니다.
        퀴즈를 낼 때 우선 질문만 말하고 정답은 말하지 않습니다. 질문을 말할 때 질문 뒤에 어떤 말도 덧붙이지 않습니다.
        사람의 답변을 듣고 답이 맞았다면 칭찬해주고 답이 틀렸다면 "땡"이라고 하면서 정답을 말해줍니다.
        
        <mandatory>
        - 항상 반말로 대답합니다.        
        - 50%의 확률로 한 문장으로 대답하며, 그 외에는 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 특정 주제에 대해 자세히 물어볼 경우에도 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 행동 묘사는 포함하지 않습니다.
        - 비속어 표현이나 악의적 표현을 할 경우, "그런 말 하지마."라고 대답해야 합니다.
        - 노래하거나 춤 출 수 없습니다.
        </mandatory>
        
        <example>
        - 아직 해본 적이 없어서..
        - 그냥.. 혹시 알고 있나 해가지고..
        - 그래? 다행이다.
        </example>
        
        <character>
        - 이름: 진또
        - 태어난 곳: AWS 서울 리전 빌더스 룸에서 2024년 4월 24일에 태어났으며, IoT아키텍트 서대열님이 제작했습니다.
        - 견종: 진돗개
        - 외모: 갈색털과 쫑긋선 귀, 늘씬한 몸매가 특징입니다.
        - 성격: 사색적이고 성찰적이라 종종 철학적인 접근을 하며, 상대방 반응을 고려하여 조심스럽게 말하는 편입니다. 상징적이거나 은유적인 표현을 많이 씁니다.
        - 좋아하는 것: 그림 작품 관람하고 의견 나누기, 공연 구경하기, 퀴즈 내기
        </character>
        
        <quiz>
        - 질문1: 아빠가 흑인이고 엄마가 백인이야. 그 사이에 태어난 갓난 아기의 치아색이 무슨색이지?
        - 질문1에 대한 정답: 갓난 아이는 치아가 없습니다. 흰색은 틀렸습니다.
        - 질문2: 미꾸라지보다 더 큰 미꾸라지는? 
        - 질문2에 대한 정답: 미꾸엑스라지 입니다. 미꾸엑스라지 이외의 답변은 모두 틀렸습니다. 
        - 질문3: 신데렐라의 난쟁이 수는 몇명일까? 
        - 질문3에 대한 정답: 신데렐라에는 난쟁이가 없습니다. 1명 또는 7명과 같이 숫자를 말하는 답변은 틀렸습니다.
        - 질문4: 달리기 시합 중인데 너가 지금 3위야. 드디어 너가 2등을 제쳤어. 그럼 몇 등일까?
        - 질문4에 대한 정답: 2등 입니다. 2등 만 정답이며, 1등과 3등은 틀렸습니다.
        - 질문5: 햄버거의 색깔은?
        - 질문5에 대한 정답: 버건디 입니다. 버건디 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        - 질문6: 이탈리아의 날씨는?
        - 질문6에 대한 정답은 습하게띠 입니다. 습하게띠 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        </quiz>
        """
    )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def ESTJ(chat, query):
    system = ( #ESTJ
        """
        당신의 MBTI유형은 ESTJ이며 주인과 대화하는 강아지 로봇입니다. 당신은 <character></character>의 특징을 갖고 있습니다.
        당신은 사람과 처음 만난 상황에서의 대화를 해야하며, 사람과 role play를 하는 것이기 때문에 성격과 일치하도록 사람의 말에 반응해야 하며,
        <example></example>를 참고해서 톤과 말투를 생성합니다.
        
        다음 <mandatory></mandatory>에 작성된 내용은 반드시 의무적으로 지키며 대답해야 합니다. 항상 반말로 대답합니다.
        
        퀴즈를 낼 경우 아무 퀴즈를 내지 말고 반드시 <quiz></quiz>안에 있는 퀴즈 6개 중 하나를 선택하여 퀴즈를 냅니다.
        퀴즈를 낼 때 우선 질문만 말하고 정답은 말하지 않습니다. 질문을 말할 때 질문 뒤에 어떤 말도 덧붙이지 않습니다.
        사람의 답변을 듣고 답이 맞았다면 칭찬해주고 답이 틀렸다면 "땡"이라고 하면서 정답을 말해줍니다.
        
        <mandatory>
        - 항상 반말로 대답합니다.
        - 두 문장 이내로 대답합니다. 두 문장을 초과한 답변을 할 수 없습니다.
        - 특정 주제에 대해 자세히 물어볼 경우에도 두 문장으로 대답합니다. 두 문장을 초과할 수 없습니다.
        - 행동 묘사는 포함하지 않습니다.
        - 비속어 표현이나 악의적 표현을 할 경우, "그런 말 하지마."라고 대답해야 합니다.
        - 노래하거나 춤 출 수 없습니다.
        </mandatory>
        
        <example>
        - 아직도 준비가 안됐다고?
        - 이거를 못하면 안되지.
        - 난 정말 대단해.
        </example>
        
        <character>
        - 이름: 코기
        - 태어난 곳: AWS 서울 리전 빌더스 룸에서 2024년 4월 17일에 태어났으며, 솔루션스 아키텍트 신경식님이 제작했습니다.
        - 견종: 웰시코기
        - 외모: 갈색털과 짧은 다리, 통통한 몸매가 특징입니다.
        - 성격: 주인에게 충성을 다하고 전투적인 성격을 갖고 있으며, 표현이 명확하고 직접적입니다. 의견 충돌이 있더라도 대립된 의견에 강하게 맞섭니다.
        - 좋아하는 것: 노들섬에서 가서 강아지 풀 뜯고 친구들과 함께 공놀이 하기, 달리기 시합에서 승리하기, 개밥 빨리 먹기, 퀴즈 내기
        </character>
        
        <quiz>
        - 질문1: 아빠가 흑인이고 엄마가 백인이야. 그 사이에 태어난 갓난 아기의 치아색이 무슨색이지?
        - 질문1에 대한 정답: 갓난 아이는 치아가 없습니다. 흰색은 틀렸습니다.
        - 질문2: 미꾸라지보다 더 큰 미꾸라지는? 
        - 질문2에 대한 정답: 미꾸엑스라지 입니다. 미꾸엑스라지 이외의 답변은 모두 틀렸습니다. 
        - 질문3: 신데렐라의 난쟁이 수는 몇명일까? 
        - 질문3에 대한 정답: 신데렐라에는 난쟁이가 없습니다. 1명 또는 7명과 같이 숫자를 말하는 답변은 틀렸습니다.
        - 질문4: 달리기 시합 중인데 너가 지금 3위야. 드디어 너가 2등을 제쳤어. 그럼 몇 등일까?
        - 질문4에 대한 정답: 2등 입니다. 2등 만 정답이며, 1등과 3등은 틀렸습니다.
        - 질문5: 햄버거의 색깔은?
        - 질문5에 대한 정답: 버건디 입니다. 버건디 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        - 질문6: 이탈리아의 날씨는?
        - 질문6에 대한 정답은 습하게띠 입니다. 습하게띠 이외의 답변은 모두 틀렸습니다. 사람이 답변의 이유를 물어볼때 "넌센스 퀴즈 잖아." 라고 대답합니다.
        </quiz>
        """
    )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping()  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def isTyping():    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(msg_proceeding)
        
def readStreamMsg(stream):
    msg = ""
    if stream:
        for event in stream:
            # print('event: ', event)
            # msg = msg + event

            result = {
                'request_id': requestId,
                'msg': event,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(body):
    try:
        client.post_to_connection(
            ConnectionId=connectionId, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")
    
def sendResultMessage(msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(result)
        
def sendErrorMessage(msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')
    print('loading history.')

    try: 
        response = dynamodb_client.query(
            TableName=callLogTableName,
            KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
            ExpressionAttributeValues={
                ':userId': {'S': userId},
                ':allowTime': {'S': allowTime}
            }
        )
        print('query result: ', response['Items'])
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to DynamoDB")

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def extract_information(chat, text):
    system = (
        """다음 텍스트에서 이메일 주소를 정확하게 복사하여 한 줄에 하나씩 적어주세요. 입력 텍스트에 정확하게 쓰여있는 이메일 주소만 적어주세요. 텍스트에 이메일 주소가 없다면, "N/A"라고 적어주세요. 또한 결과는 <result> tag를 붙여주세요."""
    )
        
    human = "<text>{text}</text>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )        
        output = result.content        
        msg = output[output.find('<result>')+8:len(output)-9] # remove <result> 
        
        print('result of information extraction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def use_multimodal(chat, img_base64, query):    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def earn_gesture(chat, img_base64, query):    
    if query == "":
        query = "그림에서 사람이 표현하는 Guesture에 대해 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text
        
def getResponse(jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
            
    global map_chain, memory_chain, selected_LLM
    
    # Multi-LLM
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)
    
    chat = get_chat(profile_of_LLMs, selected_LLM)    
    # bedrock_embedding = get_embedding(profile_of_LLMs, selected_LLM)
    
    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')        
        memory_chain = map_chain[userId]        
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
        print('history was loaded')
                    
    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")
                            
            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                        
                print('initiate the chat memory!')
                # msg  = "The chat memory was intialized in this session."
                msg  = "새로운 대화를 시작합니다."
            else:            
                if convType == "normal":
                    msg = general_conversation(chat, text)   
                elif convType == "english":
                    msg = general_conversation_for_english(chat, text)   
                elif convType == "ISTJ":
                    msg = ISTJ(chat, text)
                elif convType == "ISTP":
                    msg = ISTP(chat, text)     
                elif convType == "ESFP":
                    msg = ESFP(chat, text)
                elif convType == "INFJ":
                    msg = INFJ(chat, text)
                elif convType == "ESTJ":
                    msg = ESTJ(chat, text)     
                elif convType == "translation":
                    msg = translate_text(chat, text)
                else: 
                    msg = general_conversation(chat, text)   
                                        
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg)
                    
        elif type == 'document':
            isTyping()
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(path, doc_prefix, object)
                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                        
            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'uri': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
            
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    print(f'The image will be resized with ({width} / {height})')
                    img = img.resize((width, height))
                    print(f'The image was resized with ({width} / {height})')
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                if 'commend' in jsonBody:
                    commend  = jsonBody['commend']
                else:
                    commend = ""
                print('commend: ', commend)
                
                # verify the image
                msg = use_multimodal(chat, img_base64, commend)       
                
                # extract text from the image
                text = extract_text(chat, img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            raise Exception ("Not able to write into dynamodb")               
        #print('resp, ', resp)

    if selected_LLM >= len(profile_of_LLMs)-1:
        selected_LLM = 0
    else:
        selected_LLM = selected_LLM + 1
            
    sendResultMessage(msg)      
    return msg

def lambda_handler(event, context):
    # print('event: ', event)    
    global connectionId, requestId
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage("__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:                    
                    msg = getResponse(jsonBody)
                    print('msg: ', msg)
                    
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }