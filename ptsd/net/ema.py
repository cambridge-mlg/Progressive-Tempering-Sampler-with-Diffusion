import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995, device=None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Initialize shadow parameters on the correct device
        self.shadow = {
            name: param.clone().detach().to(self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup = {}

        # Add parameter validation
        if not 0.0 <= decay <= 1.0:
            raise ValueError("Decay must be between 0 and 1")

    def update(self, decay: float = None):
        # Allow dynamic decay rate
        decay = decay or self.decay

        with torch.no_grad():  # Ensure no gradients are tracked
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data = (
                        decay * self.shadow[name].data + (1.0 - decay) * param.data
                    )

    def apply_shadow(self):
        if not self.shadow:
            raise RuntimeError("Shadow parameters not initialized")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name].data)

    def store(self):
        with torch.no_grad():
            self._backup = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

    def restore(self):
        if not self._backup:
            raise RuntimeError("No backup parameters found")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self._backup[name])
