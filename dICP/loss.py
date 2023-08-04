import torch

class loss:
    def __init__(self, name="huber", metric=None, differentiable=True, tanh_steepness=10.0):
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
        huber_delta = self.metric
        if self.differentiable:
            # Use pseudo huber loss
            return (huber_delta**2 / (huber_delta**2 + err**2))
        else:
            return torch.where(torch.abs(err) > huber_delta, huber_delta / torch.abs(err), torch.ones_like(err))

    def __cauchy_weight(self, err):
        """
        Cauchy loss function
        """
        cauchy_delta = self.metric
        # Cauchy is differentiable by default
        return 1.0 / (1.0 + (err / cauchy_delta)**2)
    
    def __trim_weight(self, err):
        """
        Trim loss function
        """
        trim_dist = self.metric
        if len(err.shape) == 2:
            sum_dim = 1
            shape_tuple = (err.shape[0], 1)
        else:
            sum_dim = 2
            shape_tuple = (err.shape[0], err.shape[1])
        if self.differentiable:
            return 0.5 * torch.tanh(self.tanh_steepness * (trim_dist - torch.linalg.norm(err, axis=sum_dim).squeeze()) - 3.0) + 0.5
        else:
            zeros_option = torch.zeros(shape_tuple, dtype=err.dtype, device=err.device)
            ones_option = torch.ones(shape_tuple, dtype=err.dtype, device=err.device)
            return torch.where(torch.linalg.norm(err, axis=sum_dim) > trim_dist, zeros_option, ones_option)
        