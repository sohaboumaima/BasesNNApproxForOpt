import torch
from  torch.nn import *
import numpy.random as random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from math import log10, floor


torch.set_default_dtype(torch.float64)


def set_seed(seed):
    """
    Sets the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 


def from_num_to_latex(num):
    base10 = log10(abs(num))
    return "10^{" +str(floor(base10))+"}"

    
    
# def generate_Dataset_from_f(f, inputSize, num_values=200, max_val=0.9, min_val=-0.9):
def generate_Dataset_from_f(f, inputSize, num_values=200, max_val=4, min_val=-4):    
    set_seed(0)
    X = torch.Tensor(num_values, inputSize).uniform_(min_val, max_val)
    # X = torch.FloatTensor(num_values, inputSize).uniform_(min_val, max_val)
    # X = torch.Tensor([[-np.pi/2],[np.pi/2]])
    # X = torch.Tensor([[-1],[1]])
    # X = torch.Tensor([[0,1],[1,3]])
    y = [f(x) for x in X]

    return X, torch.tensor(y)


def generate_Dataset_from_f_l2_ball(f, inputSize, num_values, center, radius):    
    set_seed(0)
    X = torch.Tensor(num_values, inputSize).uniform_(-1, 1)  # Generate points in the range (-1, 1)

    ##OLD shift and scale #LSTQ OLD
    #X = torch.cat((torch.zeros(1, inputSize), X), dim=0)
    ##X = X + center
    ##shifted_X = X - X[0]
    #row_norms = torch.norm(X, dim=1)
    #max_norm = torch.max(row_norms) 
    #centered_X = radius*X / max_norm   

    ## scale and shift 
    # X = torch.cat((torch.zeros(1, inputSize), X), dim=0)
    norms = X.norm(dim=1, keepdim=True)
    normalized_X = X / norms.max()
    scaled_X = radius * normalized_X
    center_X = center.repeat(num_values,1)
    centered_X = scaled_X + center_X
    
    y = [f(x) for x in centered_X]

    return centered_X, torch.tensor(y)


class DataPoints(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class DataPointsGrad(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x.float(), y.float()
   
    def __len__(self):
        return self.len



# Norm functions
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Radial basis functions
def rbf_gaussian(x):
    return (-x.pow(2)).exp()


def rbf_linear(x):
    return x


def rbf_multiquadric(x):
    return (1 + x.pow(2)).sqrt()


def rbf_inverse_quadratic(x):
    return 1 / (1 + x.pow(2))


def rbf_inverse_multiquadric(x):
    return 1 / (1 + x.pow(2)).sqrt()

