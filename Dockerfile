FROM python:3.8.2
LABEL maintainer="NN_DRUGS_PREDICTOR"
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
CMD ["python3", "server.py"]
