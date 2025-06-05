import torch


def maybe_wrap_data_parallel(model, cfg):
    if cfg.get("use_data_parallel", False) and torch.cuda.device_count() > 1:
        try:
            # Try to clean up any existing NCCL communicators
            torch.cuda.empty_cache()
            # Set smaller buffer size for broadcast coalescing
            return torch.nn.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count()))
            )
        except RuntimeError as e:
            print(f"DataParallel initialization failed: {e}")
            print("Falling back to single GPU")
            return model
    return model


def ensure_tensor_attributes_on_device(model, device):
    """Recursively ensure all tensor attributes of the model are on the specified device."""
    for name, attr in model.__dict__.items():
        if isinstance(attr, torch.Tensor) and not name.startswith('_'):
            setattr(model, name, attr.to(device))
        elif hasattr(attr, '__dict__'):
            ensure_tensor_attributes_on_device(attr, device)


def unwrap_data_parallel(model):
    """Get the base model from DataParallel if wrapped."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def unwrap_data_parallel_model_dict(state_dict):
    """Remove 'module.' prefix from state dict keys if present (from DataParallel)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def save_model(model, opt, path, **kwargs):
    """Save model with compatibility for DataParallel."""
    # Get base model if wrapped in DataParallel
    model_to_save = unwrap_data_parallel(model)

    save_dict = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        **kwargs,
    }
    torch.save(save_dict, path)


def load_model(model, path, device):
    """Load model with compatibility for DataParallel."""
    ckpt = torch.load(path, map_location=device)
    # Load state dict into the base model, whether it's wrapped or not
    if 'model_state_dict' in ckpt:  # sometimes we save it like this
        ckpt = unwrap_data_parallel_model_dict(ckpt)
        unwrap_data_parallel(model).load_state_dict(ckpt['model_state_dict'])
    else:
        ckpt = unwrap_data_parallel_model_dict(ckpt)
        unwrap_data_parallel(model).load_state_dict(ckpt)
    return ckpt
