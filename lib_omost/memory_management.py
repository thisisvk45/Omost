import torch
import gc
from contextlib import contextmanager

high_vram = False
gpu = torch.device('cuda')
cpu = torch.device('cpu')

# Initialize GPU with pinned memory for faster transfers
torch.zeros((1, 1)).to(gpu, torch.float32)
torch.cuda.empty_cache()

models_in_gpu = []

@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return

def load_models_to_gpu(models):
    global models_in_gpu
    
    if not isinstance(models, (tuple, list)):
        models = [models]
    
    # Use set operations more efficiently
    models_set = set(models)
    models_in_gpu_set = set(models_in_gpu)
    
    models_to_remain = list(models_set & models_in_gpu_set)
    models_to_load = list(models_set - models_in_gpu_set)
    models_to_unload = list(models_in_gpu_set - models_set)
    
    if not high_vram:
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        
        models_in_gpu = models_to_remain
        
        # Aggressive memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    for m in models_to_load:
        with movable_bnb_model(m):
            # Non-blocking transfer for better performance
            m.to(gpu, non_blocking=True)
        print('Load to GPU:', m.__class__.__name__)
    
    models_in_gpu = list(models_set | set(models_to_remain))
    
    # More thorough cache clearing
    gc.collect()
    torch.cuda.empty_cache()
    
    return

def unload_all_models(extra_models=None):
    global models_in_gpu
    
    if extra_models is None:
        extra_models = []
    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]
    
    models_in_gpu = list(set(models_in_gpu + extra_models))
    
    return load_models_to_gpu([])
