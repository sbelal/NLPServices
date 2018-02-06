FROM tensorflow/tensorflow:latest-py3
LABEL author="Sal Belal"
LABEL authorEmail="Sal.Belal@visioncritical.com"

COPY ./templates /app/checkpoints
COPY ./checkpoints /app/checkpoints
COPY ./Dataset /app/Dataset
COPY ./static /app/static
COPY ./app.py /app
COPY ./FeatureExtractor.py /app
COPY ./index.html /app
COPY ./preprocess.p /app
COPY ./README.md /app
COPY ./requirements.txt /app
COPY ./SentimentAnalysisModel.py /app
COPY ./test.py /app

WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]