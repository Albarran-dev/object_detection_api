## Set up


`docker build . -t alvaro/object_detection_api`  
`docker run -it --name=object_detection_api -p=3030:80 alvaro/object_detection_api`  
Service running at http://localhost:3030/  
Docs: http://localhost:3030/docs  

From docs, or using curl/postman or similar you can upload the file and receive a json with the objects detected by the model.  
The model detects **people** and **cars**, providing the confidence score and bounding box, the results are ordered by confidence.  
The models used is **inception_resnet_v2_640x640**, hosted on the tensorflowhub.