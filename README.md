# NLPServices
NLP Services.  Each service can be run as a docker container.



docker build -t deepthought-nlp:latest .

docker run -d -p 5000:5000 deepthought-nlp:latest 


http://guido-barbaglia.blog/posts/use_docker_to_run_flask_based_rest_services.html



docker stop $(docker ps -aq);docker rm $(docker ps -aq)
docker rmi $(docker images -q)