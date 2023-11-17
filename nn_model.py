import torch
import torch.nn as nn
import math
import sympy as sym
import numpy as np
# from torch.utils.data import Dataset, DataLoader
import numpy.random as random
import os



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



def xavier_init(layer_input_size, layer_output_size):

    xavier_stddev = 1 / np.sqrt(layer_input_size + layer_output_size)
    weights = torch.tensor(np.random.randn(layer_input_size, layer_output_size) * xavier_stddev, dtype=torch.float64)
    return weights


def he_initialization(input_size, output_size):
    # Calculate the standard deviation for He initialization
    stddev = np.sqrt(2.0 / input_size)
    
    # Initialize weights with a Gaussian distribution
    weights = np.random.randn(output_size, input_size) * stddev
    
    # Initialize biases to zeros
    biases = np.zeros(output_size)
    
    return torch.tensor(weights), torch.tensor(biases)




class NNLayer(nn.Module):
  """
    An NN layer is a fully-connected linear map. It can be a hidden layer or the output layer.
    
    Arguments
        size_in:          Size of the input (u+v except for the first layer; in the first layer, size_in is equal to the number of variables in the function to recover)
        size_out:         Size of the output (number of neurons)
        activationFunct:  Activation function. Usually tanh, ReLu or sigmoid.
        index_layer:      Index of the layer in the network
  """

  def __init__(self, size_in, size_out, activationFunct, numNeurons, index_layer):
      
    super(NNLayer, self).__init__()
    
    self.size_in, self.size_out = size_in, size_out
    
    # weights = torch.Tensor(size_out, size_in) # zero values
    # weights = torch.rand(size_out, size_in) # random values

    if isinstance(activationFunct, nn.ReLU) or isinstance(activationFunct, nn.SiLU) or isinstance(activationFunct, nn.ELU):
      weights, bias = he_initialization(size_in, size_out)
    else:
      weights = xavier_init(size_out, size_in)
      bias = torch.rand(size_out) # random values

    self.weights = torch.nn.Parameter(weights)  # nn.Parameter is a Tensor and it is also a module parameter
    
    # bias = torch.Tensor(size_out) # zero values
    
    self.bias = nn.Parameter(bias)

    ## initialize weights and biases
    # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
    # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    # bound = 1 / math.sqrt(fan_in)
    # nn.init.uniform_(self.bias, -bound, bound)  # bias init

    self.activationFunct = activationFunct
    self.numNeurons = numNeurons

  def forward(self, x):
    """
    Returns the output of the layer (given the input x), which is a matrix where each row is u+2v-dimensional, except for the last layer. 
    """ 
    # If we are at the first layer, X is number of points times dimension of each point and W.T is dim of each point times size of output
    w_times_x = torch.matmul(x.double(), torch.transpose(self.weights, 0, 1)) 
    output_lin = torch.add(w_times_x, self.bias)  #T X times W.T + b (size_in times u+2v + size_in times u+2v)
    
    output = []
    # Compute the output resulting from the unary operations
    for i in range(self.numNeurons):
      func_app = self.activationFunct
      y = func_app(output_lin[:, i])
      y = y.unsqueeze(1)
      output.append(y)

    output_tens = torch.cat(tuple(output), dim=1)

    return output_tens


class NN(nn.Module):
  """
    Neural network
    
    Arguments
        inputSize:        Number of input variables in the function to recover (integer).
        outputSize:       Number of outputs returned by the network (integer).
        numLayers:        Number of (hidden and output) layers in the network (integer).
        nn_layers:        List of layers containing all of the layers in the network (including the input and output layers).
        activationFunct:  Activation function. Usually tanh or sigmoid.
        numNeurons:       Number of neurons for each hidden layer.
  """

  center = 0
  delta  = 1

  def __init__(self, inputSize, outputSize, numLayers, activationFunct, numNeurons=1):
    
    super(NN, self).__init__()
    
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.numLayers = numLayers
    self.activationFunct = activationFunct
    self.numNeurons = numNeurons

    self.nn_layers = nn.ModuleList()
    self._create_net()

  def _create_net(self):

    inp = self.inputSize

    ##self.norm_layer = nn.BatchNorm1d(inp) 

    for i in range(self.numLayers - 1):
      out = self.numNeurons

      layer = NNLayer(inp, out, self.activationFunct, self.numNeurons, i)
      # layer = nn.Linear(inp, out, bias=True)
      
      self.nn_layers.append(layer)
      inp = out

    self.lastlayer = nn.Linear(inp, self.outputSize, bias=False)
    
    # Initial value for weights from hidden to output layer
    # for param in self.lastlayer.parameters(): #T
    #    val = 0
    #    # param.data = torch.Tensor([[1,val,val,val,val]]) #T
    #    param.data = torch.Tensor([[1,val]]) #T

  def forward(self, x):

    ##x = self.norm_layer(x) 
    x = (x - self.center)/self.delta #June 2023 

    for layer in self.nn_layers:     
      x = layer(x)
      
    return self.lastlayer(x)





class NNLayer0_lstsq(nn.Module):
  """
    An NN_lstsq layer is a fully-connected linear map. It can be a hidden layer or the output layer.
    
    Arguments
        size_in:          Size of the input (u+v except for the first layer; in the first layer, size_in is equal to the number of variables in the function to recover)
        size_out:         Size of the output (number of neurons)
        index_layer:      Index of the layer in the network
  """

  def __init__(self, size_in, activationFunct, number_unary_operations, number_binary_operations, option):
      
    super(NNLayer0_lstsq, self).__init__()

    self.activationFunct = activationFunct
    self.number_unary_operations = number_unary_operations
    self.number_binary_operations = number_binary_operations
    self.option = option
    
    # What is written in the forward function only works if self.activationFunct has at most three elements    
 
    self.size_in = size_in   
    weights1 = torch.tile(torch.eye(self.size_in), (len(self.activationFunct), 1)) # stack identity matrices vertically (as many times as the number of activations in activationFunct)
    
    if self.number_binary_operations > 0:
        weights2 = torch.zeros(0,size_in) # this is used to create null row vectors to intertwine with the other row weight vectors (because we need twice as much the number of binary operations)
        
        for n1 in range(size_in-1,0,-1):
                left_matrix = torch.cat((torch.zeros(n1,size_in-1-n1), torch.ones(n1,1)), dim=1)
                left_matrix_aug = torch.cat((left_matrix,torch.zeros_like(left_matrix)),dim=1)
                left_matrix = left_matrix_aug.reshape(-1, size_in-n1)
                right_matrix = torch.eye(n1)           
                right_matrix_aug = torch.cat((torch.zeros_like(right_matrix),right_matrix),dim=1)            
                right_matrix = right_matrix_aug.reshape(-1, n1)
                join_matrix = torch.cat((left_matrix,right_matrix),dim=1)
                weights2 = torch.cat((weights2,join_matrix),dim=0)
        
        weights = torch.cat((weights1,weights2),dim=0)         
        self.weights = torch.nn.Parameter(weights)   
    
    else:
        self.weights = torch.nn.Parameter(weights1)    
        

  def forward(self, x):
    """
    Returns the output of the layer (given the input x), which is a matrix where each row is u+2v-dimensional, except for the last layer. 
    """ 
    output_lin = torch.matmul(x.double(), torch.transpose(self.weights, 0, 1)) 
    
    output = []
    
    # Compute the output resulting from the unary operations
    for i in range(self.number_unary_operations):
      if i >= 0 and i <= self.size_in-1: 
          func_app = self.activationFunct[0]
      elif len(self.activationFunct)>=2 and i >= self.size_in and i <= 2*self.size_in-1: 
          func_app = self.activationFunct[1]
      elif len(self.activationFunct)>=3 and i >= 2*self.size_in and i <= 3*self.size_in-1: 
           func_app = self.activationFunct[2]
      else: 
          print('There is something wrong in eql_1.py')
      y = func_app(output_lin[:, i])
      y = y.unsqueeze(1)
      output.append(y)
      
    if self.number_binary_operations > 0:  
        # Compute the output resulting from the binary operations
        for i in range(self.number_unary_operations, self.number_unary_operations + 2 * self.number_binary_operations, 2):
          if self.option == 2:
              y = torch.nn.functional.relu(output_lin[:, i] * output_lin[:, i+1])            
          elif self.option == 3:
              y = torch.sigmoid(output_lin[:, i] * output_lin[:, i+1])   
          elif self.option == 4:
              y = torch.tanh(output_lin[:, i] * output_lin[:, i+1]) 
          elif self.option == 5:
              y = torch.nn.functional.elu(output_lin[:, i] * output_lin[:, i+1])
          elif self.option == 6:
              y = torch.nn.functional.silu(output_lin[:, i] * output_lin[:, i+1]) 
          else: #natural basis
              y = output_lin[:, i] * output_lin[:, i+1] 
          output.append(y.unsqueeze(1))

    output_tens = torch.cat(tuple(output), dim=1)

    return output_tens


class NN_lstsq(nn.Module):
  """
    Neural network
    
    Arguments
        inputSize:        Number of input variables in the function to recover (integer).
        outputSize:       Number of outputs returned by the network (integer).
        number_unary_operations:       Number of neurons for each hidden layer.
  """

  center = 0
  delta  = 1

  def __init__(self, inputSize, option):
    
    super(NN_lstsq, self).__init__()
    
    self.inputSize = inputSize
    self.option = option
    
    def torch_pow(x):
        return 0.5*torch.pow(x,2)

    def torch_relu_pow(x):
        return torch.nn.functional.relu(0.5*torch.pow(x,2))
    
    def torch_sigmoid_pow(x):
        return torch.sigmoid(0.5*torch.pow(x,2))    

    def torch_tanh_pow(x):
        return torch.tanh(0.5*torch.pow(x,2))
    
    # What is written in the forward function of NNLayer0_lstsq only works if self.activationFunct has at most three elements     
    if self.option==1:
        self.number_binary_operations = int(self.inputSize*(self.inputSize-1)/2)
        self.activationFunct = [nn.Identity(), torch_pow] #nn.Softplus()] #torch.nn.functional.relu]
    elif self.option==7:
        self.number_binary_operations = 0        
        self.activationFunct = [nn.Identity(), torch_pow, torch.nn.functional.relu] #nn.Softplus()] #torch.nn.functional.relu]
    elif self.option >= 2 and self.option <= 6 :
        self.number_binary_operations = int(self.inputSize*(self.inputSize-1)/2)        
        self.activationFunct = [nn.Identity(), torch_pow] #nn.Softplus()] #torch.nn.functional.relu]
    elif self.option==8:
        self.number_binary_operations = 0        
        self.activationFunct = [nn.Identity(), torch_pow, torch.sigmoid]
    elif self.option==9:
        self.number_binary_operations = 0         
        self.activationFunct = [nn.Identity(), torch_pow, torch.tanh]
    elif self.option==10:
        self.number_binary_operations = 0         
        self.activationFunct = [nn.Identity(), torch_pow, torch.nn.functional.elu]
    elif self.option==11:
        self.number_binary_operations = 0         
        self.activationFunct = [nn.Identity(), torch_pow, torch.nn.functional.silu]

  
    self.number_unary_operations = len(self.activationFunct)*self.inputSize

    self.nn_layers = nn.ModuleList()
    self._create_net()

  def _create_net(self):

    inp = self.inputSize

    out = self.number_unary_operations

    layer0 = NNLayer0_lstsq(inp,self.activationFunct,self.number_unary_operations,self.number_binary_operations, self.option)    
    self.nn_layers.append(layer0)
    
    inp = out + self.number_binary_operations

    self.lastlayer = nn.Linear(inp, 1, bias=True)
    

  def forward(self, x):

    x = (x - self.center)/self.delta #June 2023  

    for layer in self.nn_layers:     
      x = layer(x) 

    return self.lastlayer(x)

  def compute_activations(self,x):

    x = (x - self.center)/self.delta #June 2023  
      
    for layer in self.nn_layers:     
        x = layer(x) 
        
    return x

  def set_weights_last_layer(self,weights):
      for idx, param in enumerate(self.lastlayer.parameters()):
          if idx == 0:
              param.data = weights[1:].reshape(1,-1)
          elif idx == 1:
              param.data[0] = weights[0]
          else:
              print('There is something wrong')



class RBF_interp(nn.Module):


  center = 0
  delta  = 1
   
  def __init__(self, inputSize, radial_function, num_kernels=None) -> None:
    super(RBF_interp, self).__init__()
    self.inputsize = inputSize
    self.radial_function = radial_function

    if num_kernels is None:
       self.num_kernels = 2*self.inputsize + 1
    else:
       self.num_kernels = num_kernels

    self.parameters = torch.zeros(self.num_kernels,)

    self.kernels = torch.zeros(self.num_kernels, self.inputsize)
    nn.init.uniform_(self.kernels, a=-1.0, b=1.0)

    self.linear_params = torch.zeros(self.inputsize+1,)

  
  def forward(self, x):
     
    x = (x - self.center)/self.delta #June 2023
       
    batch_size = x.size(0)

    c = self.kernels.expand(batch_size, self.num_kernels, self.inputsize)

    diff = x.view(batch_size, 1, self.inputsize) - c

    r = torch.norm(diff, 2, dim=-1)

    rbfs = self.radial_function(r)

    out = self.weights.expand(batch_size, 1,
                                self.num_kernels) * rbfs.view(
                                    batch_size, 1, self.num_kernels)
    
    fpar = self.linear_params[:self.inputsize].t()
    
    lin = fpar.expand(batch_size, self.inputsize) * x
    
    #Add the linear term
    s = out.sum(dim=-1) + lin.sum() + self.linear_params[self.inputsize]

    
    return s

  
  def interpolate(self, trainloader):
    
    # Find the right about of point; option 1: choose randomly num_kernels points

    for X, y in trainloader:

      self.kernels = (X - self.center)/self.delta
      self.num_kernels = self.kernels.shape[0]
      n = X.size(0)
      m = self.inputsize
      
     
      kernel_norms = torch.norm(self.kernels[:, None] - self.kernels, dim=2)

      # Apply your radial_function element-wise
      Q = self.radial_function(kernel_norms)

      # Create P matrix
      P = torch.cat((self.kernels, torch.ones(n, 1)), 1).t()

      # Construct the system matrix [Q P; P^T 0]
      A = torch.cat((torch.cat((Q, P.t()), 1), torch.cat((P, torch.zeros(m+1, m+1)), 1)), 0)

      # Create the right-hand side vector [f(Y); 0]
      b = torch.cat((y, torch.zeros(m+1,)))

    x = torch.linalg.lstsq(A, b.unsqueeze(1)).solution

    # Extract 'a' and 'b' from the solution
    self.weights = x[:n].t()
    self.linear_params = x[n:]



    

     


  



