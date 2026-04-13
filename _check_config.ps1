cd D:\Issac\polyarb_lab
py -3 -c "from src.config_runtime.loader import load_runtime_config; cfg = load_runtime_config('config/settings.yaml'); print(type(cfg)); print(cfg.model_fields.keys() if hasattr(cfg,'model_fields') else dir(cfg))"
