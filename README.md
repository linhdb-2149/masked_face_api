## Create virtualenv
```python
    virtualenv -p python3 env
    source env/bin/activate
    pip install -r requirements.txt
```

## Config serving directory and port
Configure HOST + PORT in *configs.py*  
Configure direction absolute path in *model_cf.config*; change PORT variable if modify in *configs.py*

## Model
Download model, place serving folder in directory
```
https://drive.google.com/file/d/1jdTFcqdKRTA-4Bg_os-Ai9n-ZTm0hPOi/view?usp=sharing
```

## Run Serving
```
    tensorflow_model_server --port=8100 --rest_api_port=8101 --model_config_file=absolute_path/model_cf.config

```

## Start server
```
    python app.py
```

