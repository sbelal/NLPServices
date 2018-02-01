FROM tensorflow/tensorflow:latest-py3
LABEL author="Sal Belal"
LABEL authorEmail="Sal.Belal@visioncritical.com"

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]

