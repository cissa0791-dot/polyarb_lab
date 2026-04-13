cd D:\Issac\polyarb_lab
py -3 -c "from src.config_runtime.loader import load_runtime_config; cfg = load_runtime_config('config/settings.yaml'); md = cfg.market_data; print(type(md)); print(md)"
