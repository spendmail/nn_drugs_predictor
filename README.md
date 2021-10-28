## Requirements

- [x] python 3.8.*

## Installation

```sh
sudo apt update
sudo apt install python3-virtualenv
virtualenv -p /usr/bin/python3 venv
./venv/bin/python3 -m pip install -r requirements.txt
```

## Launching

```sh
./venv/bin/python3 predict_local_name.py
```

## Training

```sh
./venv/bin/python3 train_internal_names.py
```