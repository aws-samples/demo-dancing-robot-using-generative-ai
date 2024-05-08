import boto3
import os
import time
import re
import base64
import boto3
import uuid
import json
import traceback
import copy    

from botocore.config import Config
from PIL import Image
from io import BytesIO
from urllib import parse
from multiprocessing import Process, Pipe

s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_photo_prefix = os.environ.get('s3_photo_prefix')
path = os.environ.get('path')
endpoint_name = 'sam-endpoint-2024-04-10-01-35-30'

profile_of_Image_LLMs = json.loads(os.environ.get('profile_of_Image_LLMs'))
selected_LLM = 0

seed = 43
cfgScale = 7.5
# height = 1152
# width = 768

enableParallel = True
k = 3

smr_client = boto3.client("sagemaker-runtime")
s3_client = boto3.client('s3')   
rekognition_client = boto3.client('rekognition')
  
def get_client(profile_of_Image_LLMs, selected_LLM):
    profile = profile_of_Image_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
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
    return boto3_bedrock, modelId

def img_resize(image):
    imgWidth, imgHeight = image.size 
    
    max_length = 1024

    if imgWidth < imgHeight:
        imgWidth = int(max_length/imgHeight*imgWidth)
        imgWidth = imgWidth-imgWidth%64
        imgHeight = max_length
    else:
        imgHeight = int(max_length/imgWidth*imgHeight)
        imgHeight = imgHeight-imgHeight%64
        imgWidth = max_length 

    image = image.resize((imgWidth, imgHeight), resample=0)
    return image

def show_faces(bucket, key):                          
    image_obj = s3_client.get_object(Bucket=bucket, Key=key)
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    print(f"(original) width: {width}, height: {height}, size: {width*height}")
    
    img = img_resize(img)
    
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=100)
    val = buffer.getvalue()

    response = rekognition_client.detect_faces(Image={'Bytes': val},Attributes=['ALL'])

    imgWidth, imgHeight = img.size       
    ori_image = copy.deepcopy(img)

    for faceDetail in response['FaceDetails']:
        print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) 
              + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')

        box = faceDetail['BoundingBox']
        left = imgWidth * box['Left']
        top = imgHeight * box['Top']
        width = imgWidth * box['Width']
        height = imgHeight * box['Height']

        print(f"imgWidth : {imgWidth}, imgHeight : {imgHeight}")
        print('Left: ' + '{0:.0f}'.format(left))
        print('Top: ' + '{0:.0f}'.format(top))
        print('Face Width: ' + "{0:.0f}".format(width))
        print('Face Height: ' + "{0:.0f}".format(height))

    return ori_image, imgWidth, imgHeight, int(left), int(top), int(width), int(height), response

def show_labels(img_path, target_label=None):
    if target_label is None:
        Settings = {"GeneralLabels": {"LabelInclusionFilters":[]},"ImageProperties": {"MaxDominantColors":1}}
        print(f"target_label_None : {target_label}")
    else:
        Settings = {"GeneralLabels": {"LabelInclusionFilters":[target_label]},"ImageProperties": {"MaxDominantColors":1}}
        print(f"target_label : {target_label}")
    
    box = None
    
    image = Image.open(img_path).convert('RGB')
    image = img_resize(image)

    buffer = BytesIO()
    image.save(buffer, format='jpeg', quality=100)
    val = buffer.getvalue()
    
    response = rekognition_client.detect_labels(Image={'Bytes': val},
        MaxLabels=15,
        MinConfidence=0.7,
        # Uncomment to use image properties and filtration settings
        Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"],
        Settings=Settings
    )

    imgWidth, imgHeight = image.size       
    ori_image = copy.deepcopy(image)
    color = 'white'

    for item in response['Labels']:
        # print(item)
        if len(item['Instances']) > 0:
            print(item)
            print(item['Name'], item['Confidence'])

            for sub_item in item['Instances']:
                color = sub_item['DominantColors'][0]['CSSColor']
                box = sub_item['BoundingBox']
                break
        break
    try:
        left = imgWidth * box['Left']
        top = imgHeight * box['Top']
        width = imgWidth * box['Width']
        height = imgHeight * box['Height']

        print(f"imgWidth : {imgWidth}, imgHeight : {imgHeight}")
        print('Left: ' + '{0:.0f}'.format(left))
        print('Top: ' + '{0:.0f}'.format(top))
        print('Object Width: ' + "{0:.0f}".format(width))
        print('Object Height: ' + "{0:.0f}".format(height))
        return ori_image, imgWidth, imgHeight, int(left), int(top), int(width), int(height), color, response
    except:
        print("There is no target label in the image.")
        return _, _, _, _, _, _, _, _, _

def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

def decode_image(img):
    img = img.encode("utf8") if type(img) == "bytes" else img
    buff = BytesIO(base64.b64decode(img))
    image = Image.open(buff)
    return image

def invoke_endpoint(endpoint_name, payload):
    response = smr_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Accept="application/json",
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    data = response["Body"].read().decode("utf-8")
    return data

def encode_image(image, formats="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=formats)
    img_str = base64.b64encode(buffer.getvalue())
    return img_str

def generate_outpainting_image(boto3_bedrock, modelId, object_img, mask_img, text_prompt):
    body = json.dumps({
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": text_prompt,              # Optional
            # "negativeText": negative_prompts,    # Optional
            "image": image_to_base64(object_img),      # Required
            # "maskPrompt": mask_prompt,               # One of "maskImage" or "maskPrompt" is required
            "maskImage": image_to_base64(mask_img),  # Input maskImage based on the values 0 (black) or 255 (white) only
        },                                                 
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            # "quality": "standard",
            "cfgScale": cfgScale,
            # "height": height,
            # "width": width,
            "seed": seed
        }
    })
            
    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json", 
        contentType="application/json"
    )
    print('response: ', response)
            
    # Output processing
    response_body = json.loads(response.get("body").read())
    img_b64 = response_body["images"][0]
    print(f"Output: {img_b64[0:80]}...")
    
    return img_b64

def parallel_process(conn, boto3_bedrock, modelId, object_img, mask_img, text_prompt, object_name, object_key):    
    img_b64 =  generate_outpainting_image(boto3_bedrock, modelId, object_img, mask_img, text_prompt)
            
    # upload
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=object_key,
        ContentType='image/jpeg',
        Body=base64.b64decode(img_b64)
    )
    print('response: ', response)
            
    url = path+s3_photo_prefix+'/'+parse.quote(object_name)
    print('url: ', url)
    
    conn.send(url)
    conn.close()
                    
def lambda_handler(event, context):
    global selected_LLM
        
    print(event)
    
    start_time_for_generation = time.time()
    
    jsonBody = json.loads(event['body'])
    print('request body: ', json.dumps(jsonBody))
    
    requestId = jsonBody["requestId"]
    print('requestId: ', requestId)
    bucket = jsonBody["bucket"]   
    key = jsonBody["key"]   
    
    url_original = path+parse.quote(key)
    print('url_original: ', url_original)
    
    # mask
    ext = key.split('.')[-1]
    if ext == 'jpg':
        ext = 'jpeg'
    
    target_label = None  
        
    if target_label == None:
        object_image, width, height, f_left, f_top, f_width, f_height, human_res = show_faces(bucket, key) ## detect_faces        
    #else:
    #    object_image, width, height, f_left, f_top, f_width, f_height, color, human_res = show_labels(bucket, key, target_label=target_label)
    
    encode_object_image = encode_image(object_image,formats=ext.upper()).decode("utf-8")
    inputs = dict(
        encode_image = encode_object_image,
        input_box = [f_left, f_top, f_left+f_width, f_top+f_height]
    )

    predictions = invoke_endpoint(endpoint_name, inputs)
    mask_image = decode_image(json.loads(predictions)['mask_image'])

    object_img = img_resize(object_image)
    mask_img = img_resize(mask_image)
    id = uuid.uuid1()
    
    if enableParallel==False: # single 
        object_name = f'photo_{id}.{ext}'
        outpaint_prompt = 'forrest'
        text_prompt = f'a human with a {outpaint_prompt[0]} background'
        
        boto3_bedrock, modelId = get_client(profile_of_Image_LLMs, selected_LLM)
        
        img_b64 = generate_outpainting_image(boto3_bedrock, modelId, object_img, mask_img, text_prompt)
                                
        # upload
        object_key = f'{s3_photo_prefix}/{object_name}'  # MP3 파일 경로
        print('object_key: ', object_key)
        
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=object_key,
            ContentType='image/jpeg',
            Body=base64.b64decode(img_b64)
        )
        print('response: ', response)
            
        end_time_for_generation = time.time()
        time_for_photo_generation = end_time_for_generation - start_time_for_generation
        
        url_generated = path+s3_photo_prefix+'/'+parse.quote(object_name)
        print('url_generated: ', url_generated)
        
        selected_LLM = selected_LLM + 1
        if selected_LLM == len(profile_of_Image_LLMs):
            selected_LLM = 0
        
        result = {            
                "url_original": url_original,
                "url_generated": url_generated,
                "time_taken": str(time_for_photo_generation)
        }
        print('result: ', result)
        
    else: # multiple (k)
        generated_urls = []    
        processes = []
        parent_connections = []
        
        outpaint_prompt =['sky','building','forest']   # ['desert', 'sea', 'mount']
        
        for i in range(k):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            
            boto3_bedrock, modelId = get_client(profile_of_Image_LLMs, selected_LLM)
            text_prompt =  f'a human with a {outpaint_prompt[i]} background'
            
            object_name = f'photo_{id}_{i+1}.{ext}'
            object_key = f'{s3_photo_prefix}/{object_name}'  # MP3 파일 경로
            print('object_key: ', object_key)
        
            process = Process(target=parallel_process, args=(child_conn, boto3_bedrock, modelId, object_img, mask_img, text_prompt, object_name, object_key))
            processes.append(process)
            
            selected_LLM = selected_LLM + 1
            if selected_LLM == len(profile_of_Image_LLMs):
                selected_LLM = 0
                        
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            url = parent_conn.recv()
            generated_urls.append(url)

        for process in processes:
            process.join()
                
        end_time_for_generation = time.time()
        time_for_photo_generation = end_time_for_generation - start_time_for_generation
        
        print('generated_urls: ', json.dumps(generated_urls))
        
        result = {            
                "url_original": url_original,
                "url_generated": json.dumps(generated_urls),
                "time_taken": str(time_for_photo_generation)
        }
        print('result: ', result)
        
    return {
        "isBase64Encoded": False,
        'statusCode': 200,
        'body': json.dumps(result)
    }
