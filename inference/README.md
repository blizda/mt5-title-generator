For inference, just go to ```docker``` dir, specify you W&B api key in ```.env``` file and run ```docker-compose up```(completely service starting may be really long due to model downloading and onnx optimisation)

After this, docker-compose raise web-service on ```127.0.0.1:8080/model``` 

For getting model response, just sent post-request with 
field ```message```, with news text inside it