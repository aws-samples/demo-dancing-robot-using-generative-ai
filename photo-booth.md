# Generative Image Generation for Photo Booth 

In Photo Booth, it takes photos of visitors and separates their faces and backgrounds to create new images. 

The image is generated through the following process: 

1) Take a photo using Booth's Pad and send it to the server. 

2) Use Rekognition to identify the location (bounding box) of the face. 

3) Use the SAM configured with SageMaker Endpoint to extract only the face. 

4) Use Bedrock's Titan image generator to create a new image.

## Image upload

To send large files, we use a presigned URL. 

1) Request a Presigned URL. The CloudFront address as seen from the client is "dxt1m1ae24b28.cloudfront.net" (the URL may change). Request a Presigned URL in the following manner: 

```text
POST https://dxt1m1ae24b28.cloudfront.net/upload
{
  "type": "photo",
  "filename": "andy_portrait_2.jpg",
  "contentType": "image/jpeg"
}
```

The result at this time is as follows. Here, the UploadURL is valid for 5 minutes. 

```text
{
    "statusCode": 200,
    "body": "{\"Bucket\":\"storage-for-demo-dansing-robot-533267442321-ap-northeast-2\",\"Key\":\"photo/andy_portrait_2.jpg\",\"Expires\":300,\"ContentType\":\"image/jpeg\",\"UploadURL\":\"https://storage-for-demo-dansing-robot-533267442321-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/docs/andy_portrait_2.jpg?Content-Type=image%2Fjpeg&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAXYKJXNKIRO3ZJZN4%2F20240410%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20240410T224428Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAMv49uyZaGs4FJ3e7NPv3vwVUntkkeVSub3SDKw1eEL4AiA9O%2F6aImNfebK6mxDZvYboSrJ9Ba%2B7BchSqczM0SnNRSqiAwg4EAAaDDUzMzI2NzQ0MjMyMSIMSSHU5cg2k5mSE7KSKv8CeGozybV1giKOi3%2F2SFqUHZuZ%2FwKQgx2SOXkszLUZEUq66ZONMjjjewCn3PiG%2BHFNEc9nqSXVjsPWIb2vRkKG27nwInJF36SibN0qejMI8c9br8KatqHqYAinnduQhrspI3TEJJ0sqF11HZ7odW4eYKZxrofdrod00FeUesSNA%2BI5eCYL7yPEytEViYTeCK%2Fyy7VIS%2FBcGG9bkZhxjgu4gifzUoJm4qll0HjB2prqidtaECI3VcmHHJma13Lhv9ATYo%2BGQtpaxOftl0IJKDEYwRxtxd3pO3%2FlCfqthxbP%2Bx2jHs9lLDiazmekyl4ReU2GJ%2B7bKpFmt2UMRysFjw0aylniq0aEumuH9vnShlzHn5cSLcBCx0K3Dl2DJYR2adPrX2Br4NQUzaNuB9sLqDStYjLNGvy7wwytG6Y3gmfLCXyOttKaTzGP%2F8G&X-Amz-Signature=e8294d3304d4ed60872a4826732777337f1f98a561&X-Amz-SignedHeaders=host\"}"
}
````

2) Send the file via HTTP PUT to the UploadURL. 


```java
var xmlHttp = new XMLHttpRequest();
xmlHttp.open("PUT", uploadURL, true);       
const blob = new Blob([input.files[0]], { type: contentType });
```

3) Request image generation. At this time, use the CloudFront domain and '/photo' API for the request URL. 
   
```text   
POST  https://dxt1m1ae24b28.cloudfront.net/photo
{
    "requestId": "b123456abc",
    "bucket": "storage-for-demo-dansing-robot-533267442321-ap-northeast-2",
    "key": "photo/andy_portrait_2.jpg"
}
```

The result at this time is as follows. The prefix "photo_" is added to the name of the uploaded file, creating a new name. If multiple files are generated, "_1", "_2", "_3" and so on are added. 

```java
{
    "url_original": "https://dxt1m1ae24b28.cloudfront.net/photo/andy_portrait_2.jpg",
    "url_generated": "[\"https://dxt1m1ae24b28.cloudfront.net/photo/photo_507ff273-f8df-11ee-8f9b-69f7819ad4a8_1.jpeg\", \"https://dxt1m1ae24b28.cloudfront.net/photo/photo_507ff273-f8df-11ee-8f9b-69f7819ad4a8_2.jpeg\", \"https://dxt1m1ae24b28.cloudfront.net/photo/photo_507ff273-f8df-11ee-8f9b-69f7819ad4a8_3.jpeg\"]",
    "time_taken": "26.989280939102173"
}
```

Examples of generated images are as follows.

#### Original Image

<img src="https://github.com/kyopark2014/demo-ai-dansing-robot/blob/main/photo-booth/andy_portrait_2.jpg" width="800">

#### Generated image 

Currently, the Titan image generator can only be used in the Virginia and Oregon regions, so I have generated 3 images. 


![photo_86db851a-f8dd-11ee-872f-c90482ae9123_1](https://github.com/kyopark2014/demo-ai-dansing-robot/assets/52392004/2ee958f9-8292-4147-8cee-5f2a51920850)

![photo_86db851a-f8dd-11ee-872f-c90482ae9123_2](https://github.com/kyopark2014/demo-ai-dansing-robot/assets/52392004/c2b3dd10-fb01-47ef-9586-24d581779d21)

![photo_86db851a-f8dd-11ee-872f-c90482ae9123_3](https://github.com/kyopark2014/demo-ai-dansing-robot/assets/52392004/770a9d66-b054-49a7-8ff7-aaaea6d1155b)

![photo_86db851a-f8dd-11ee-872f-c90482ae9123_4](https://github.com/kyopark2014/demo-ai-dansing-robot/assets/52392004/a0c6e6b3-08d8-493d-a4cc-1e019ffa030a)
