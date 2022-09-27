FROM python:3.9

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

RUN mkdir app
WORKDIR /app

RUN mkdir /.cache
RUN chmod -R 777 /.cache
