# This is the abstract class experiment that will contain the main functions that are used in all experiments
import torch
from  torch.nn import *
import math
from torch.utils.data import DataLoader, random_split, Subset
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
from optperfprofpy import calc_perprof, draw_simple_pp
import pandas as pd
import pycutest

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)
# import warnings
# warnings.filterwarnings("error")


class Experiment:


    FUNCTION_SET1 = {
    0: 'ARGLINA',
    1: 'ARGTRIGLS',
    2: 'ARWHEAD',
    3: 'BDEXP',
    4: 'BOXPOWER',
    5: 'BROWNAL',
    6: 'COSINE',
    7: 'CURLY10',
    8: 'DIXON3DQ',
    9: 'DQRTIC',
    10: 'ENGVAL1',
    11: 'EXTROSNB',
    12: 'FLETBV3M',
    13: 'FLETCBV3',
    14: 'FLETCHBV',
    15: 'FLETCHCR',
    16: 'FREUROTH',
    17: 'INDEFM',
    18: 'MANCINO',
    19: 'MOREBV',
    20: 'NONCVXU2',
    21: 'NONCVXUN',
    22: 'NONDIA',
    23: 'NONDQUAR',
    24: 'PENALTY2',
    25: 'POWER',
    26: 'QING',
    27: 'QUARTC',
    28: 'SENSORS',
    29: 'SINQUAD',
    30: 'SCOSINE',
    31: 'SCURLY10',
    32: 'SPARSINE',
    33: 'SPARSQUR',
    34: 'SSBRYBND',
    35: 'TRIDIA',
    36: 'TRIGON1',
    37: 'TOINTGSS'
    } 


    # Problems with fixed dimension from Clement Luis Math. Programming
    FUNCTION_SET2 = {\
    0: 'ALLINITU',\
    1: 'BARD',\
    2: 'BIGGS6',\
    3: 'BOX3',\
    4: 'BROYDN7D',\
    5: 'BRYBND',\
    6: 'CUBE',\
    7: 'DENSCHND',\
    8: 'DENSCHNE',\
    9: 'DIXMAANA1',\
    10: 'DIXMAANB',\
    11: 'DIXMAANC',\
    12: 'DIXMAAND',\
    13: 'DIXMAANE1',\
    14: 'DIXMAANF',\
    15: 'DIXMAANG',\
    16: 'DIXMAANH',\
    17: 'DIXMAANI1',\
    18: 'DIXMAANJ',\
    19: 'DIXMAANK',\
    20: 'DIXMAANL',\
    21: 'ENGVAL2',\
    22: 'ERRINROS',\
    23: 'EXPFIT',\
    24: 'FMINSURF',\
    25: 'GROWTHLS',\
    26: 'GULF',\
    27: 'HAIRY',\
    28: 'HATFLDD',\
    29: 'HATFLDE',\
    30: 'HEART6LS',\
    31: 'HEART8LS',\
    32: 'HELIX',\
    33: 'HIELOW',\
    34: 'HIMMELBB',\
    35: 'HIMMELBG',\
    36: 'HUMPS',\
    37: 'KOWOSB',\
    38: 'LOGHAIRY',\
    39: 'MARATOSB',\
    40: 'MEYER3',\
    41: 'MSQRTALS',\
    42: 'MSQRTBLS',\
    43: 'OSBORNEA',\
    44: 'OSBORNEB',\
    45: 'PENALTY3',\
    46: 'SNAIL',\
    47: 'SPMSRTLS',\
    48: 'STRATEC',\
    49: 'VIBRBEAM',\
    50: 'WATSON',\
    51: 'WOODS',\
    52: 'YFITU'
    } 

    SIZE_DATASET_EXTRA = 100 # Extra points to evaluate the predicted function also outside the bounds
    MAX_BOUND_EXTRA = 10
    MIN_BOUND_EXTRA = -10
    RADIUS = 1
    GRAD_SIMPLEX = False # Only if grad_FD==True; In addition to FD gradients, use the simplex gradient to populate the dataset used for gradient learning 
    COMPUTE_HESS = True
    GRAD_FD = False


    def __init__(self, activ_functions={'ReLU':torch.nn.ReLU(), 'Sigmoid': torch.sigmoid, 'Tanh':torch.tanh}, plot=True):

        self.afunc = activ_functions
        self.dimensions = [20, 40, 60]
        self.FDToll_NN1 = 1e-4
        self.plot = plot
        self.save_plot = True

    
    def create_problem(self, string_function, FDtollerance, dim, batch_size_percent=0.05):

        if string_function == 'ARGLINA' or string_function == 'ARGLALE':                
                cutest_prob = pycutest.import_problem(string_function, sifParams={'N':dim, 'M':2*dim})
        # Problems with variable dimension
        elif string_function in self.FUNCTION_SET1.values(): 
                print('\nFunction currently tested:',string_function)
                cutest_prob = pycutest.import_problem(string_function, sifParams={'N':dim})		
        elif string_function == 'BROYDN7D':
            dim_negcurv_probs = {'BROYDN7D': 5}
            cutest_prob = pycutest.import_problem(string_function, sifParams={'N/2':dim_negcurv_probs[string_function]})	
            inputSize_val = cutest_prob.n 	
            print('\nFunction currently tested:',string_function,', Dim: ',inputSize_val)
        elif string_function == 'BRYBND':
            dim_negcurv_probs = {'BRYBND': 10}
            cutest_prob = pycutest.import_problem(string_function, sifParams={'N':dim_negcurv_probs[string_function]})		
            inputSize_val = cutest_prob.n 
            print('\nFunction currently tested:',string_function,', Dim: ',inputSize_val)
        elif string_function in self.FUNCTION_SET2.values():
            cutest_prob = pycutest.import_problem(string_function)
            inputSize_val = cutest_prob.n        		
            print('\nFunction currently tested:',string_function,', Dim: ',inputSize_val)
        else:
            raise('SOMETHING WRONG - string_function: ',string_function)

        
        def fun_aux(x):        
            if x.requires_grad:
                x = torch.squeeze(x).detach().numpy()
            else:
                x = torch.squeeze(x).numpy()         
            out = cutest_prob.obj(x,gradient=False)
            out = torch.Tensor([out]) #+ 0.5*torch.randn(1)
            return out

        func = fun_aux  
        self.func = func
        self.cutest_prob = cutest_prob
        inputSize_val = cutest_prob.n 
        
        # Create the dataset
        x0 = cutest_prob.x0
        center = torch.tensor(x0, dtype=torch.float64).reshape(1,-1)  
        size_dataset = math.ceil(((inputSize_val+1)*(inputSize_val+2)/2)/0.8)
        X, y = generate_Dataset_from_f_l2_ball(func, inputSize=inputSize_val, num_values=size_dataset, center=center, radius=self.RADIUS)        
        dataset = DataPoints(X, y)

        
        # Random split of the dataset
        train_set_size = int(len(dataset) * 0.8)
        test_set_size = len(dataset) - train_set_size


        train_set, test_set = random_split(dataset, [train_set_size, test_set_size])

        number = math.ceil(batch_size_percent*((inputSize_val+1)*(inputSize_val+2)/2))
        power = math.ceil(math.log2(number))
        batch_size = min(64, max(2 ** power, 2))

        if inputSize_val == 20:
            batch_size = 16
        elif inputSize_val == 40:
            batch_size = 32
        elif inputSize_val == 60:
            batch_size = 64


        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_set, batch_size=test_set_size, shuffle=True)

        #Compute the max_norm based on the training dataloader

        delta = 0

        for X_batch,_ in train_dataloader:
            batch_max_norm = torch.norm(X_batch - center, p=2, dim=1).max()
            delta = max(delta, batch_max_norm.item())

        
        # Create the extra dataset to evaluate the predicted function outside the bounds
        X_extra, y_extra = generate_Dataset_from_f(func, inputSize=inputSize_val, num_values=self.SIZE_DATASET_EXTRA, max_val=self.MAX_BOUND_EXTRA, min_val=self.MIN_BOUND_EXTRA)
        dataset_extra = DataPoints(X_extra, y_extra)
        
        dataloader_extra = DataLoader(dataset=dataset_extra, batch_size=len(dataset_extra), shuffle=True)


        print('\n\n\n ******',string_function,', n =',inputSize_val)#', Number of gradients ',len(gradients))


        # Dataset for gradient interpolation
        # Points used to compute simplex gradients
        train_set_size_simplex = math.floor(1*(inputSize_val+1)*(inputSize_val+2)/2)
        train_set_simplex, _ = random_split(dataset, [train_set_size_simplex, len(dataset) - train_set_size_simplex])
        train_dataloader_simplex = DataLoader(dataset=train_set_simplex, batch_size=1, shuffle=True)
        
        points_grad, gradients, hessians = self.create_gradients(train_dataloader_simplex, func, FDtollerance, inputSize_val)
        
        print('\n\n\n ******',string_function,', n =',inputSize_val)#', Number of gradients ',len(gradients))
        # print(gradients)
        
        dataset_grad = DataPoints(torch.transpose(torch.cat(points_grad, 1), 0, 1), torch.transpose(torch.cat(gradients, 1), 0, 1))
        
        train_set_grad_size = int(len(dataset_grad) * 1)
        train_set_grad, _ = random_split(dataset_grad, [train_set_grad_size, 0])        
        
        grad_dataloader = DataLoader(dataset=train_set_grad, batch_size=train_set_grad_size, shuffle=True, drop_last=True)       

        

        # Dataset for Hessian
        if self.COMPUTE_HESS:
        
            dataset_hess = DataPoints(torch.transpose(torch.cat(points_grad, 1), 0, 1), hessians)
                    
            train_set_hess_size = int(len(dataset_hess) * 1)
            train_set_hess, _ = random_split(dataset_hess, [train_set_hess_size, 0])        

            self.hess_dataloader = DataLoader(dataset=train_set_hess, batch_size=train_set_grad_size, shuffle=True, drop_last=True)



        return train_dataloader, test_dataloader, dataloader_extra, grad_dataloader, inputSize_val, center, delta
    

    def create_gradients(self, train_dataloader_simplex, func, FDtollerance, inputSize_val):
        "Generate n+1 simplex gradients for each point in the dataset"
        
        points_grad  = []
        gradients    = []
        hessians     = []
                
        if self.GRAD_FD:
            
            h = FDtollerance #math.sqrt(np.finfo(float).eps)
            I = torch.eye(inputSize_val)
            
            x_temp = {}
            
            for X, y in train_dataloader_simplex:
                
                x = X[0].reshape(-1,1)
                
                g = torch.zeros_like(x)
            
                if self.GRAD_SIMPLEX:
                    g_simplex = {}
                    for i in range(inputSize_val+1):
                        g_simplex[i] = torch.zeros_like(x)
                    A_simplex = [] # contains points y-y0
                    A_simplex.append(torch.transpose(x,0,1))
                    b_simplex = [] # contains f(y)
                    b_simplex.append(y[0].unsqueeze(-1))
        
                for i in range(inputSize_val):
        
                    x_temp[i] = x + h * I[:,i:i+1]
                    fh = func(x_temp[i])
                    
                    g[i] = (fh - y[0])/h
                    
                    if self.GRAD_SIMPLEX:
                        A_simplex.append(torch.transpose(x_temp[i],0,1))
                        b_simplex.append(torch.Tensor([fh]))  
                
                if self.GRAD_SIMPLEX:
                    A_simplex = torch.cat(A_simplex)
                    b_simplex = torch.cat(b_simplex).reshape(-1,1)
                    for i in range(inputSize_val+1):
                        A_temp = A_simplex - A_simplex[i]
                        A = A_temp[[j for j in range(A_temp.shape[0]) if j != i], :]
                        Delta = torch.max(torch.linalg.vector_norm(A, ord=2, dim=1, keepdim=True))
                        A = A/Delta
                        b_temp = b_simplex - b_simplex[i]
                        b = b_temp[[j for j in range(b_temp.shape[0]) if j != i], :]
                        g_simplex[i] = torch.linalg.solve(A, b)/Delta
                        
                
                points_grad.append(x.clone().detach())
                gradients.append(g) 
                
                if self.GRAD_SIMPLEX:
                    for i in range(inputSize_val):
                        points_grad.append(x_temp[i].clone().detach())
                        gradients.append(g_simplex[i+1].reshape(-1,1))  # we skip i==0, which is associated with self.x, because the FD gradient is more accurate than the simplex gradient
        
        else:
            for X, y in train_dataloader_simplex:
                
                x = X[0].reshape(-1,1)
        
                g = self.get_grad_obj_function_cutest(x, self.cutest_prob)
                
                    
                points_grad.append(x.clone().detach())
                gradients.append(g) 

                if self.COMPUTE_HESS:
                    hess = self.get_hess_obj_function_cutest(x, self.cutest_prob, inputSize_val)                       
                    hessians.append(hess) 
                        
        return points_grad, gradients, hessians


    def get_grad_obj_function_cutest(self, x, cutest_prob):
          if x.requires_grad:
            x = torch.squeeze(x).detach().numpy()  
          else:
            x = torch.squeeze(x).numpy()         
          _, g = cutest_prob.obj(x,gradient=True)
          g = torch.Tensor(g)
          return g.reshape(1,-1)


    def get_hess_obj_function_cutest(self, x, cutest_prob, inputSize_val):
          if x.requires_grad:
            x = torch.squeeze(x).detach().numpy()  
          else:
            x = torch.squeeze(x).numpy()         
          hess = cutest_prob.hess(x)
          hess = torch.Tensor(hess)
          return hess.reshape(inputSize_val,-1)
    
    def loss_function(self, X, y, model):
        
        loss_fn = torch.nn.MSELoss()
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        
        # Add regularization
        ## l2 norm
        # l_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        ## l1 norm
        l_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters()) # torch.norm(p, 1)
        reg_val = 0
        loss = loss + l_norm*reg_val 
        
        return loss
    
    def error_function(self, X, y, model):        
        pred = model(X)
        err = torch.abs(pred - y.unsqueeze(-1))/torch.abs(y.unsqueeze(-1))
        return err.sum()/X.shape[0]

    def get_objective(self, fvals_dict, tau, numevals_dict=None):

        #fvals_dict is a dictionary having the algorithms as keys and for each algorithm there is an inner dictionary having problems as keys'
        methods_list = []
        problems_list = []    
        min_val = {}
        obj     = []
        start_val = {}

        firs_ele_dic = list(fvals_dict.values())[0]
                
        for idx_prob, func in firs_ele_dic.items():
        
            start_val[idx_prob] = func[0]
            
            min_val_dict = {}
            min_val_dict[0] = min(func)
    
            for idx_alg in fvals_dict:
                if idx_alg != 0:
                    min_val_dict[idx_alg] = min(fvals_dict[idx_alg][idx_prob])
    
            min_val[idx_prob] = min(min_val_dict.values()) # minimum value among the minimum values achived by the algorithms on problem idx
    
        for idx_prob in range(len(firs_ele_dic)):
    
            cut = min_val[idx_prob] + tau * (start_val[idx_prob] - min_val[idx_prob]) # cutoff threshold
    
            res = {}
    
            for idx_alg in fvals_dict:
                res[idx_alg] = list(filter(lambda i: i <= cut, fvals_dict[idx_alg][idx_prob])) # The filter() method filters the given sequence with the help of a function that tests each element in the sequence to be true or not
                if not res[idx_alg]:
                    obj.append(np.nan)
                else:
                    if numevals_dict is None:
                        obj.append(list(fvals_dict[idx_alg][idx_prob]).index(res[idx_alg][0]))
                    else:
                        numeval = numevals_dict[idx_alg][idx_prob]
                        ind = fvals_dict[idx_alg][idx_prob].index(res[idx_alg][0])
                        obj.append(numeval[ind])
                problems_list.append(idx_prob)
                methods_list.append(idx_alg)

        # change all the elements equal to 0 to 1 to avoid division by zero when computing performance profiles            
        obj = [1 if x == 0 else x for x in obj]
    
        return obj, problems_list, methods_list
        

    
    def draw_perf_profiles(self, objective_fvals_dict, tau, lns=None, filename=None, numevals_dict=None, dataframe_ready=None):

        if dataframe_ready is None:

            objective_vals_list0, problems_list, methods_list = self.get_objective(objective_fvals_dict, tau, numevals_dict)
            objective_vals = pd.Series(objective_vals_list0, dtype=float, name='obj')
            # max_ratio = objective_vals.max()
            objective_vals_list = objective_vals#.fillna(2*objective_vals.max())
            problems_col = problems_list
            problems = pd.Series(problems_col, dtype=int, name='problem')
            methods_col = methods_list
            methods = pd.Series(methods_col, dtype=str, name='method')
            objective_vals_col = objective_vals_list
            objective_vals = pd.Series(objective_vals_col, dtype=float, name='obj')
            
            # Create dataframe for performance profiles
            dataframe = pd.DataFrame([problems, methods, objective_vals]).T
        else:
            dataframe = pd.read_csv(dataframe_ready)
        
        taus, solver_vals, solvers, _, max_ratio = calc_perprof(dataframe, ['problem'], ['obj'], ['method'])
        # plot_title_pp = 'Dim = ' + str(self.inputSize_val) + ', Bound = ' + self.string_bound + ', |Y| = ' + str(self.size_dataset)
        plot_title_pp = '$\\tau =' + from_num_to_latex(tau) + '$'
        draw_simple_pp(taus, solver_vals, solvers, plot_title_pp, max_ratio, filename, lns)
        return dataframe
    

    # This function needs to be implemented in all other sections
    def run_exp(self):
        pass
        # 
    


                








