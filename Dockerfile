FROM tensorflow/tensorflow:latest-py3
MAINTAINER Sal Belal "sbelal@gmail.com"

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]

