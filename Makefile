tf_serve:
	tensorflow_model_server --port=8100 --rest_api_port=8101 --model_config_file=/home/doan.bao.linh/Desktop/Project/MaskDetector/mask_api/model_cf.config

check:
	saved_model_cli show --dir  /home/doan.bao.linh/Desktop/Project/MaskDetector/mask_api/serving/mask/1 --all
