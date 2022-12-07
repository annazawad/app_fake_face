# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /flask_deploy

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]