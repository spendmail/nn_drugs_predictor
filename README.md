## Launch Server in docker container

```sh
docker-compose up -d
```

## Send HTTP request

```sh
curl --header "Content-Type: application/json" \
--request POST \
--data '{"name":"АНАЛЬГИН ТАБ. 0.5Г N20 ОБН"}' \
http://localhost:5000/ 
```

## Local Installation

```sh
sudo apt update
sudo apt install python3-virtualenv
virtualenv -p /usr/bin/python3 venv
./venv/bin/python3 -m pip install -r requirements.txt
```

## Training

```sh
./venv/bin/python3 train.py
```

## Usage in CLI

```sh
./venv/bin/python3 predict.py "АНАЛЬГИН ТАБ. 0.5Г N20 ОБН"
```

## Launch a local web server

```sh
./venv/bin/python3 server.py
```
