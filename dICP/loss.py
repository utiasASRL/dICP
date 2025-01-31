import torch

class loss:
    def __init__(self, name="huber", metric=1.0, differentiable=False, tanh_steepness=10.0):
        self.name = name        # Options are "huber", "cauchy", "trim" 
                                # (Trim isn't a real loss but is grouped here for convinience)
        self.metric = metric    # Metric used for loss function
        self.differentiable = differentiable    # Whether to use differentiable loss function
        self.tanh_steepness = tanh_steepness    # Steepness of tanh function for soft losses
    
    def get_weight(self, err):
        if self.name == "huber":
            return self.__huber_weight(err)
        elif self.name == "cauchy":
            return self.__cauchy_weight(err)
        elif self.name == "trim":
            return self.__trim_weight(err)
        else:
            raise ValueError("Invalid loss name: {}".format(self.name))
        
    def __huber_weight(self, err):
        """
        Huber loss function
        """
        if len(err.shape) == 2: sum_dim = 1
        else: sum_dim = 2
        err_norm = torch.linalg.norm(err, axis=sum_dim)
        if self.differentiable:
            # Use pseudo huber loss
            return (self.metric**2 / (self.metric**2 + err_norm**2))
        else:
            return torch.where(err_norm > self.metric, self.metric / err_norm, torch.ones_like(err_norm))

    def __cauchy_weight(self, err):
        """
        Cauchy loss function
        """
        if len(err.shape) == 2: sum_dim = 1
        else: sum_dim = 2
        # Cauchy is differentiable by default
        return 1.0 / (1.0 + (torch.linalg.norm(err, axis=sum_dim) / self.metric)**2)
    
    def __trim_weight(self, err):
        """
        Trim loss function
        """
        if len(err.shape) == 2:
            sum_dim = 1
            shape_tuple = (err.shape[0], 1)
        else:
            sum_dim = 2
            shape_tuple = (err.shape[0], err.shape[1])
        if self.differentiable:
            return 0.5 * torch.tanh(self.tanh_steepness * (self.metric - torch.linalg.norm(err, axis=sum_dim)) - 3.0) + 0.5
        else:
            zeros_option = torch.zeros(shape_tuple, dtype=err.dtype, device=err.device)
            ones_option = torch.ones(shape_tuple, dtype=err.dtype, device=err.device)
            return torch.where(torch.linalg.norm(err, axis=sum_dim) < self.metric, ones_option, zeros_option)
        