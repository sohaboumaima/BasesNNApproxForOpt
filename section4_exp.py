import torch
from FullLowEval import FullLowEvalOpt
from  torch.nn import *
import numpy.random as random
import numpy as np
import os

from experiment import Experiment
import pycutest
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

## This file runs the experiments in section 3 


def set_seed_(seed):
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

    
set_seed_(0)

class Experiment_sect4(Experiment):


    def __init__(self, activ_functions={}, options=range(6), plot=True):
        super().__init__(activ_functions, plot)
        self.options = options
        self.feval_dict = {}
        self.func_dict  = {}
        self.reset_data()

    
    def reset_data(self):
        for op in self.options:
            self.feval_dict[op] = {}
            self.func_dict[op]  = {}


    def run_exp(self, functions=Experiment.FUNCTION_SET1, dim=0):

        for idx in functions:

            string_function = functions[idx]

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
            

            x0 = torch.tensor(cutest_prob.x0, dtype=torch.float64).reshape(-1,1)

            for op in self.options:
                set_seed_(idx)
        
                fleopt_nosurr = FullLowEvalOpt(x0.clone().detach(), string_function, cutest_prob, init_step = 'FE', option=op)
                try:
                    fleopt_nosurr.Full_Low_Eval()
                    self.feval_dict[op][idx] = fleopt_nosurr.fevalhist
                    self.func_dict[op][idx] = fleopt_nosurr.fvalhist
                except Exception as e:
                    self.feval_dict[op][idx] = [float('nan')]
                    self.func_dict[op][idx] = [float('nan')]            
                    print('\n**********\n ERROR - FUNCTION: ',string_function,' - Error message: ',str(e),'\n**********')
        
        


            
    

if __name__ == "__main__":
# 
    exp = Experiment_sect4()

    lns = {'FLE': '-', 'FLE-S Natural Basis': '-.', 'FLE-S Sigmoid Basis': '--', 'FLE-S RBF Basis':'-.', 'FLE-S NN ReLU':'--', 'FLE-S NN SiLU':'-.'}

    for dim in [20, 40, 60]:
        exp.run_exp(exp.FUNCTION_SET1, dim = dim)

        # Save exp.feval_dict and exp.func_dict

        with open(os.getcwd() + '/data/FLE_set1_dim_'+ str(dim) +'_feval_NN', 'wb') as handle:
            pickle.dump(exp.feval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        with open(os.getcwd() + '/data/FLE_set1_dim_'+ str(dim) +'_func_NN_', 'wb') as handle:
            pickle.dump(exp.func_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        # Tolerance 1e-2
        dataframe_1e2 = exp.draw_perf_profiles(exp.func_dict, 1e-2, lns=lns, filename= "PP_dim" +str(dim) + "_m2_noS_SN_SS_38probs.png", numevals_dict=exp.feval_dict)

        # Tolerance 1e-5
        dataframe_1e5 = exp.draw_perf_profiles(exp.func_dict, 1e-5, lns=lns, filename= "PP_dim" +str(dim) + "_m5_noS_SN_SS_38probs.png", numevals_dict=exp.feval_dict)\

    exp.reset_data()

    exp.run_exp(exp.FUNCTION_SET2)


    with open(os.getcwd() + '/data/FLE_set2_feval_NN', 'wb') as handle:
        pickle.dump(exp.feval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    with open(os.getcwd() + '/data/FLE_set2_func_NN', 'wb') as handle:
        pickle.dump(exp.func_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    # Tolerance 1e-2
    dataframe_1e2 = exp.draw_perf_profiles(exp.func_dict, 1e-2, lns=lns, filename= "PP_dim00_m2_noS_SN_SS_negcurv.png", numevals_dict=exp.feval_dict)

    # Tolerance 1e-5
    dataframe_1e5 = exp.draw_perf_profiles(exp.func_dict, 1e-5, lns=lns, filename= "PP_dim00_m5_noS_SN_SS_negcurv.png", numevals_dict=exp.feval_dict)\


