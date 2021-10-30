## Requirements

- [x] python 3.8.*

## Installation

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

## Launch in CLI

```sh
./venv/bin/python3 predict.py
```

## Launch a web server

```sh
./venv/bin/python3 server.py 
```

## Query to a web server

```sh
curl --header "Content-Type: application/json" \
--request POST \
--data '{"name":"АНАЛЬГИН ТАБ. 0.5Г N20 ОБН"}' \
http://127.0.0.1:5000/
```