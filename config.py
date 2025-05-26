def get_config():
    return{
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
        
    }
    
def get_weights_path(config,epoch):
    return f"{config['model_folder']}/{config['model_basename']}{epoch}.pt"