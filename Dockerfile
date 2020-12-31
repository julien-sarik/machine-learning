FROM python:3.8.7-alpine3.12

RUN apk --update add --no-cache g++
RUN pip install pandas
