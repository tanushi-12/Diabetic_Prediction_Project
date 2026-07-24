
import os
 
from src.config import MODEL_DIR
 

TRANSFORMER_MODELS = {
    "FT-Transformer": "FT-Transformer_checkpoint",
    "TabTransformer": "TabTransformer_checkpoint",
}
 
 
def is_transformer_model(model_name):
    return model_name in TRANSFORMER_MODELS
 
 
def transformer_checkpoint_path(model_name):
    return os.path.join(MODEL_DIR, TRANSFORMER_MODELS[model_name])
 
 
def transformer_available(model_name):
    """Only list a transformer as usable if its checkpoint folder actually exists."""
    return os.path.isdir(transformer_checkpoint_path(model_name))
 
 
def load_transformer_model(model_name):
   
    import torch
    import pytorch_lightning as pl
    from pytorch_tabular import TabularModel
 
    original_torch_load = torch.load
    original_trainer_init = pl.Trainer.__init__
 
    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)
 
    def _cpu_trainer_init(self, *args, **kwargs):
        kwargs["accelerator"] = "cpu"
        if "devices" in kwargs:
            kwargs["devices"] = 1       
        if "gpus" in kwargs:
            kwargs["gpus"] = None       
        return original_trainer_init(self, *args, **kwargs)
 
    torch.load = _cpu_load
    pl.Trainer.__init__ = _cpu_trainer_init
    try:
        return TabularModel.load_model(transformer_checkpoint_path(model_name), map_location="cpu")
    finally:
        torch.load = original_torch_load
        pl.Trainer.__init__ = original_trainer_init
 
 
def predict_with_transformer(model, input_df, target_column):
   
    pred_df = model.predict(input_df)
 
    preds = pred_df[f"{target_column}_prediction"].values
 
    proba_cols = sorted([
        c for c in pred_df.columns
        if c.startswith(f"{target_column}_") and c.endswith("_probability")
    ])
    probas = pred_df[proba_cols].values
 
    return preds, probas
