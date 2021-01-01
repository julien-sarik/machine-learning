# FROM python:3.8.7-alpine3.12
FROM python:3

# RUN apk --update add --no-cache g++
RUN pip install pandas sklearn
