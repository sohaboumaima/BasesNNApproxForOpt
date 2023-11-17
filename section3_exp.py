import torch
from  torch.nn import *
import numpy.random as random
from nn_model import NN, NN_lstsq, RBF_interp
from torch.utils.data import DataLoader
from utils import DataPoints, rbf_gaussian
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from experiment import Experiment
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


class Experiment_sect3(Experiment):

    NUM_LAYERS = 3

    def __init__(self, activ_functions={'NN ReLU':torch.nn.ReLU(), 'NN ELU': torch.nn.ELU(), 'NN SiLU': torch.nn.SiLU(), 'NN Sigmoid': torch.sigmoid, 'NN Tanh':torch.tanh}, epochs=300, options=[5, 6, 10, 11], plot=True): #'NN ReLU':torch.nn.ReLU(),
        super().__init__(activ_functions, plot)
        self.epochs = epochs
        self.lstsq_options = options
        self.table_latex_dict_lstsq = {}
        self.table_latex_dict_NN = {}


    def create_problem(self, string_function, FDtollerance, dim):
        train_dataloader, test_dataloader, dataloader_extra, grad_dataloader, inputSize_val, center, delta = super().create_problem(string_function, FDtollerance, dim, 1)

        # we are just interested in the center point, so all the datasets have just one point
        f_center = torch.tensor([self.func(center[0])])
        
        center_dataset = DataPoints(center, f_center) # dataset of just one point 
        center_loader = DataLoader(dataset=center_dataset, batch_size=1, shuffle=True)
        
        center_point, center_grad, center_hess = self.create_gradients(center_loader, self.func, FDtollerance, inputSize_val)             

        center_grad_dataset = DataPoints(torch.transpose(torch.cat(center_point, 1), 0, 1), center_grad)            
        self.grad_dataloader_center = DataLoader(dataset=center_grad_dataset, batch_size=1, shuffle=True, drop_last=True)
        grad_dataloader = self.grad_dataloader_center

        if self.COMPUTE_HESS:
            center_hess_dataset = DataPoints(torch.transpose(torch.cat(center_point, 1), 0, 1), center_hess)    
            self.hess_dataloader = DataLoader(dataset=center_hess_dataset, batch_size=1, shuffle=True, drop_last=True)
        
        return train_dataloader, test_dataloader, dataloader_extra, grad_dataloader, inputSize_val, center, delta


    def rbf_opt(self, surrogate_model, train_dataloader, grad_dataloader=None, get_grad_surrogate_point=None):

        surrogate_model.interpolate(train_dataloader)

        avg_norm_diff = {}  
                 
        def compare_function_values(grad_dataloader, surrogate_model, get_grad_surrogate_point):
                            
            for X_grad, _ in grad_dataloader: 

                func_true = torch.Tensor([self.func(X_grad[0])])
                func_surr = surrogate_model(torch.Tensor(X_grad[0]).reshape(1,-1))
                avg_func_norm_diff = torch.mean(torch.norm(func_true - func_surr, p=2, dim=-1)/torch.max(torch.norm(func_true,p=2,dim=-1),torch.norm(func_surr, p=2, dim=-1))) #torch.Tensor([0.01])))
                avg_true_func_norm = torch.mean(torch.norm(func_true, p=2, dim=-1))
                avg_surr_func_norm = torch.mean(torch.norm(func_surr, p=2, dim=-1))
                print('Avg Func Difference Norm: ',avg_func_norm_diff.item(),', Avg Surr Func Norm: ',avg_surr_func_norm.item(),', Avg True Func Norm: ',avg_true_func_norm.item())
                                    
            return avg_func_norm_diff.item()
        
        avg_norm_diff[0] = compare_function_values(grad_dataloader, surrogate_model, get_grad_surrogate_point)            
        fig = plt.figure()
        
        def compare_gradients(grad_dataloader, surrogate_model, get_grad_surrogate_point):
            
            for X_grad, y_grad in grad_dataloader: 
                # y_grad is dim_sample times n    
                grad_surr = get_grad_surrogate_point(surrogate_model, X_grad)                
                avg_grad_norm_diff = torch.mean(torch.norm(y_grad - grad_surr, p=2, dim=-1)/torch.max(torch.norm(y_grad,p=2,dim=-1),torch.norm(grad_surr, p=2, dim=-1))) #torch.Tensor([0.01])))
                # print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item())
                
                avg_true_grad_norm = torch.mean(torch.norm(y_grad, p=2, dim=-1))
                # print('Avg True Grad Norm: ',avg_true_grad_norm.item())
                
                avg_surr_grad_norm = torch.mean(torch.norm(grad_surr, p=2, dim=-1))
                print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item(),', Avg Surr Grad Norm: ',avg_surr_grad_norm.item(),', Avg True Grad Norm: ',avg_true_grad_norm.item())
                
            return avg_grad_norm_diff.item() 
        
        avg_norm_diff[1] = compare_gradients(grad_dataloader, surrogate_model, get_grad_surrogate_point)
        
        
        def compare_hessians(surrogate_model):
            
            for X_hess, y_hess in self.hess_dataloader: 

                X_hess_surr = self.get_hess_surrogate_point(surrogate_model, X_hess)
                
                avg_hess_norm_diff = torch.mean(torch.norm(y_hess - X_hess_surr, p=2, dim=-1)/torch.max(torch.norm(y_hess,p=2,dim=-1),torch.norm(X_hess_surr, p=2, dim=-1)))  # torch.Tensor([0.01])))                
                avg_true_hess_norm = torch.mean(torch.norm(y_hess, p=2, dim=-1))
                avg_surr_hess_norm = torch.mean(torch.norm(X_hess_surr, p=2, dim=-1))
                print('Avg Hess Difference Norm: ',avg_hess_norm_diff.item(),', Avg Surr Hess Norm: ',avg_surr_hess_norm.item(),', Avg True Hess Norm: ',avg_true_hess_norm.item())

            return avg_hess_norm_diff.item() 

        if self.COMPUTE_HESS:            
            avg_norm_diff[2] = compare_hessians(surrogate_model)            
        
        return avg_norm_diff, surrogate_model




    
    def lstsq_opt(self, surrogate_model, train_dataloader, grad_dataloader=None, get_grad_surrogate_point=None):
        A_lstsq = []
        b_lstsq = []
        
        # Training data
        for X, y in train_dataloader:
            A_lstsq.append(torch.cat((torch.ones(X.shape[0],1),surrogate_model.compute_activations(X)),dim=1))
            b_lstsq.append(y)
        A_lstsq = torch.cat(A_lstsq)
        b_lstsq = torch.cat(b_lstsq).reshape(-1,1)
        print('Number of points: ',A_lstsq.shape[0])
        
        def compute_reg_lstsq(A,b): 
            tol_svd        = np.finfo(float).eps**5
            U, Sdiag, V    = torch.linalg.svd(A, full_matrices=False)
            indices        = Sdiag < tol_svd      
            Sdiag[indices] = tol_svd
            Sinv           = torch.diag(1/Sdiag)
            x              = torch.matmul(V.T,torch.matmul(Sinv,torch.matmul(U.T,b)))
            return x


        weights_lstsq = compute_reg_lstsq(A_lstsq,b_lstsq)
        
        surrogate_model.set_weights_last_layer(weights_lstsq)
        loss_fn = torch.nn.MSELoss()
        out_loss = loss_fn(torch.matmul(A_lstsq,weights_lstsq), b_lstsq)

        out_none = torch.Tensor([0])
        
        print('Train Residual: ',out_loss.item(),', Test Residual: ',out_none.item())
        
        out_aux6 = [] 
        out_aux6.append(out_loss.detach())
        

        avg_norm_diff = {}  
                 
        def compare_function_values(grad_dataloader, surrogate_model, get_grad_surrogate_point):
                            
            for X_grad, _ in grad_dataloader: 

                func_true = torch.Tensor([self.func(X_grad[0])])
                func_surr = surrogate_model(torch.Tensor(X_grad[0]).reshape(1,-1))
                avg_func_norm_diff = torch.mean(torch.norm(func_true - func_surr, p=2, dim=-1)/torch.max(torch.norm(func_true,p=2,dim=-1),torch.norm(func_surr, p=2, dim=-1))) #torch.Tensor([0.01])))
                avg_true_func_norm = torch.mean(torch.norm(func_true, p=2, dim=-1))
                avg_surr_func_norm = torch.mean(torch.norm(func_surr, p=2, dim=-1))
                print('Avg Func Difference Norm: ',avg_func_norm_diff.item(),', Avg Surr Func Norm: ',avg_surr_func_norm.item(),', Avg True Func Norm: ',avg_true_func_norm.item())
                                    
            return avg_func_norm_diff.item()
        
        avg_norm_diff[0] = compare_function_values(grad_dataloader, surrogate_model, get_grad_surrogate_point)            
        fig = plt.figure()
        
        def compare_gradients(grad_dataloader, surrogate_model, get_grad_surrogate_point):
            
            for X_grad, y_grad in grad_dataloader: 
                # y_grad is dim_sample times n    
                grad_surr = get_grad_surrogate_point(surrogate_model, X_grad)                
                avg_grad_norm_diff = torch.mean(torch.norm(y_grad - grad_surr, p=2, dim=-1)/torch.max(torch.norm(y_grad,p=2,dim=-1),torch.norm(grad_surr, p=2, dim=-1))) #torch.Tensor([0.01])))
                # print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item())
                
                avg_true_grad_norm = torch.mean(torch.norm(y_grad, p=2, dim=-1))
                # print('Avg True Grad Norm: ',avg_true_grad_norm.item())
                
                avg_surr_grad_norm = torch.mean(torch.norm(grad_surr, p=2, dim=-1))
                print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item(),', Avg Surr Grad Norm: ',avg_surr_grad_norm.item(),', Avg True Grad Norm: ',avg_true_grad_norm.item())
                
            return avg_grad_norm_diff.item() #grad_loss.item()
        
        avg_norm_diff[1] = compare_gradients(grad_dataloader, surrogate_model, get_grad_surrogate_point)
        
        
        def compare_hessians(surrogate_model):
            
            for X_hess, y_hess in self.hess_dataloader: 

                X_hess_surr = self.get_hess_surrogate_point(surrogate_model, X_hess)
                
                avg_hess_norm_diff = torch.mean(torch.norm(y_hess - X_hess_surr, p=2, dim=-1)/torch.max(torch.norm(y_hess,p=2,dim=-1),torch.norm(X_hess_surr, p=2, dim=-1)))  # torch.Tensor([0.01])))                
                avg_true_hess_norm = torch.mean(torch.norm(y_hess, p=2, dim=-1))
                avg_surr_hess_norm = torch.mean(torch.norm(X_hess_surr, p=2, dim=-1))
                print('Avg Hess Difference Norm: ',avg_hess_norm_diff.item(),', Avg Surr Hess Norm: ',avg_surr_hess_norm.item(),', Avg True Hess Norm: ',avg_true_hess_norm.item())

            return avg_hess_norm_diff.item() #grad_loss.item()

        if self.COMPUTE_HESS:            
            avg_norm_diff[2] = compare_hessians(surrogate_model)            
        
        return out_aux6, avg_norm_diff, surrogate_model


    def draw_histogram(self, file, dic=None, method=1):

        if dic is None:
            with open(file, 'rb') as handle:
                table_latex_dict = pickle.load(handle)
        else:
            table_latex_dict = dic

        count_plot = 1
        
        plt.figure(figsize=(8, 5))
        plt.subplots_adjust(hspace=1.0)

        for idx_func_grad_hess in [0,1,2]: # 0: func, 1: grad, 2: hess 
            threshold = 5
    
            values_dict = {} # {option: list of values of |H - nabla^2 f| over all the problems}
            values_mod_dict = {} # like values_dict but all values over the threshold are cut.
            # normalized_colors_dict = {} # dict that contains the color for each problem 
    
            # Loop through the options ('key' is for option, 'idx' is for problem)
            for key in table_latex_dict[idx_func_grad_hess][0].keys():        
                
                values_dict[key] = [table_latex_dict[idx_func_grad_hess][idx][key] for idx in table_latex_dict[idx_func_grad_hess].keys()]
                values_mod_dict[key] = [item if item < threshold else threshold for item in values_dict[key]]  
                                 
        
        # for idx_func_grad_hess in [0,1,2]: # 0: func, 1: grad, 2: hess
            categories = list(table_latex_dict[idx_func_grad_hess][0].keys())
            ax = plt.subplot(3, 1, count_plot)
            
            # Create an empty list to store the boxplot artists
            boxplots = []
            positions = [0, 1, 2, 3, 4]
            i = 0
            # Iterate over the categories
            for category in categories:

                # positions.append(i)

                # Create a boxplot and add it to the list
                boxplot = ax.boxplot(values_mod_dict[category], positions=[positions[i]],
                                     showmeans=False, medianprops={'color': 'red'},
                                     flierprops={'markersize': 1, 'marker': 'o', 'markerfacecolor': 'blue', 'markeredgecolor': 'blue'})
                boxplots.append(boxplot)
                i += 1
            
            # Set the x-axis tick positions and labels
            ax.set_xticks(positions)
            # categories_x = [item+1 for item in positions]
            ax.set_xticklabels(categories)
            
            # Add labels and title
            # if idx_func_grad_hess == 2:
            if method:
                plt.xlabel('Bases',fontsize=12)
            
            # plt.ylabel(r'${0}$'.format(ylabel_dict[idx_func_grad_hess]),fontsize=8)
            
                if idx_func_grad_hess == 0:
                    plt.ylabel(r'$|\hat{m} - f|$',fontsize=12)
                elif idx_func_grad_hess == 1:
                    plt.ylabel(r'$\Vert \nabla \hat{m} - \nabla f \Vert$',fontsize=12)
                elif idx_func_grad_hess == 2:                        
                    plt.ylabel(r'$\Vert \nabla^2 \hat{m} - \nabla^2 f \Vert$',fontsize=12)
            
            else:
                if idx_func_grad_hess == 0:
                    plt.ylabel(r'$|\hat{f}_{NN} - f|$',fontsize=12)
                elif idx_func_grad_hess == 1:
                    plt.ylabel(r'$\Vert \nabla \hat{f}_{NN} - \nabla f \Vert$',fontsize=12)
                elif idx_func_grad_hess == 2:                        
                    plt.ylabel(r'$\Vert \nabla^2 \hat{f}_{NN} - \nabla^2 f \Vert$',fontsize=12)
                
            
            plt.ylim([0, 1.2])
            
            count_plot += 1
                
        plt.plot()

        

        figure_filename = file + '_combined.png'
        plt.savefig(figure_filename)
        plt.close()

    def get_grad_surrogate_point(self, model, xx):
        'Compute the gradient of the surrogate model at each of the points in the batch xx'
        
        g_rows = []
        for i in range(xx.shape[0]): 
            y = torch.Tensor(xx[i]).reshape(1,-1) #torch.transpose(xx, 0, 1) 
            y.requires_grad_()
            pred = model(y)
            g_rows.append(torch.autograd.grad(pred, y, create_graph=True)[0])
        g = torch.stack(g_rows)
        
        return torch.transpose(g, 0, 1)
    

    def get_hess_surrogate_point(self, model, xx):
        'Compute the Hessian of the surrogate model at each of the points in the batch xx'
        
        g_rows = []
        for i in range(xx.shape[0]):    
            y = torch.Tensor(xx[i]).reshape(1,-1) #torch.transpose(xx, 0, 1) 
            y.requires_grad_()
            pred = model(y)
            grads = torch.autograd.grad(pred, y, create_graph=True, retain_graph=True)[0]
            hessians = []
            for grad in grads:
                hessian = []
                for g in grad:
                    hess = torch.autograd.grad(g, y, retain_graph=True)[0]
                    hessian.append(hess)
                hessians.append(torch.stack(hessian))
            # g_rows.append(torch.stack(hessians))
            g_rows.append(hessians[0].permute(2, 0, 1).squeeze())
        hess = torch.stack(g_rows).squeeze(2)
        return hess
    

    def SGD_opt_with_grad(self, model, learning_rate, train_dataloader, test_dataloader, grad_dataloader, get_grad_surrogate_point, dataloader_extra):
        
        loss_values = []
        loss_test_values = []
    
        loss_values_epoch = []
        loss_test_values_epoch = []
    
        err_values = []
        err_test_values = []
        
        err_values_epoch = []
        err_test_values_epoch = []
        err_extra_values_epoch = []
                
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)

        print("---------------------------------------------------------")
        
        start_time = time.time()

        best_loss_val = float('inf')
        
        for epoch in range(self.epochs):
            it = 0
            running_loss = 0
            running_error = 0
            model.train()
            for X, y in train_dataloader:
                
                loss = self.loss_function(X, y, model)
                loss_values.append(loss.item()) 
                running_loss += loss.item()
                loss_print = loss.item()
                
                err = self.error_function(X, y, model)
                err_values.append(err.item())
                running_error += err.item()

                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Compute the gradient
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                  
                it += 1

            # Save loss and error for the whole epoch
            loss_values_epoch.append(running_loss/it)
            err_values_epoch.append(running_error/it)
            # Compute the testing loss
            loss_test_tot = 0
            err_test_tot = 0
            it = 0
            model.eval()

            with torch.no_grad():
                for X_test, y_test in test_dataloader:
                    loss_test = self.loss_function(X_test, y_test, model)
                    loss_test_tot += loss_test.item()
                    err_test = self.error_function(X_test, y_test, model)
                    err_test_tot += err_test.item()
                    it+= 1
                loss_test_values.append(loss_test_tot)
                err_test_values.append(err_test_tot)
                loss_test_values_epoch.append(loss_test_tot/it)
                err_test_values_epoch.append(err_test_tot/it)
                validation_loss = loss_test_tot/it

                if best_loss_val > validation_loss:

                    best_loss_val = validation_loss

                    # save the new model
                    best_model_dict = model.state_dict()

                # Compute the error on the extra dataset
                err_extra_tot = 0
                it = 0
                for X_extra, y_extra in dataloader_extra:
                    err_extra = self.error_function(X_extra, y_extra, model)
                    err_extra_tot += err_extra.item()
                    it += 1
                if err_extra_tot:
                    err_extra_values_epoch.append(err_extra_tot/it)
                else:
                    err_extra_values_epoch.append(err_extra_tot)
                
            model.train()

            # Update the learning rate scheduler based on validation loss
            scheduler.step(validation_loss)
            running_time = time.time() - start_time
            print('------ Epoch: ',epoch,' - Iter: ',it,' - Training loss: ',loss_print,' - Testing loss: ',loss_test_values[len(loss_test_values)-1],' - Error: ',err.item(),' - Testing Error: ',err_test_tot,' - Time: ',running_time)   
               
        print("Training Complete")
        print("Total number of trainable parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("---------------------------------------------------------")

        best_model = NN(model.inputSize, model.outputSize, model.numLayers, model.activationFunct, model.numNeurons)

        best_model.center = model.center
        best_model.delta  = model.delta

        best_model.load_state_dict(best_model_dict)

        model = best_model

        
        avg_norm_diff = {}      

            
        def compare_function_values(grad_dataloader, surrogate_model, get_grad_surrogate_point):
            
            for X_grad, _ in grad_dataloader: 

                func_true = torch.Tensor([self.func(X_grad[0])])
                func_surr = surrogate_model(torch.Tensor(X_grad[0]).reshape(1,-1))
                avg_func_norm_diff = torch.mean(torch.norm(func_true - func_surr, p=2, dim=-1)/torch.max(torch.norm(func_true,p=2,dim=-1),torch.norm(func_surr, p=2, dim=-1)))
                avg_true_func_norm = torch.mean(torch.norm(func_true, p=2, dim=-1))
                avg_surr_func_norm = torch.mean(torch.norm(func_surr, p=2, dim=-1))
                print('Avg Func Difference Norm: ',avg_func_norm_diff.item(),', Avg Surr Func Norm: ',avg_surr_func_norm.item(),', Avg True Func Norm: ',avg_true_func_norm.item())
                                    
            return avg_func_norm_diff.item() #grad_loss.item()
        
        avg_norm_diff[0] = compare_function_values(self.grad_dataloader_center, model, get_grad_surrogate_point)            
        
        
        def compare_gradients(grad_dataloader, surrogate_model, get_grad_surrogate_point):
            
            for X_grad, y_grad in grad_dataloader: 

                # y_grad is dim_sample times n                    
                avg_grad_norm_diff = torch.mean(torch.norm(y_grad - get_grad_surrogate_point(surrogate_model, X_grad), p=2, dim=-1)/torch.max(torch.norm(y_grad,p=2,dim=-1),torch.norm(get_grad_surrogate_point(surrogate_model, X_grad), p=2, dim=-1))) #torch.Tensor([0.01])))
                # print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item())
                
                avg_true_grad_norm = torch.mean(torch.norm(y_grad, p=2, dim=-1))
                # print('Avg True Grad Norm: ',avg_true_grad_norm.item())
                
                avg_surr_grad_norm = torch.mean(torch.norm(get_grad_surrogate_point(surrogate_model, X_grad), p=2, dim=-1))
                print('Avg Grad Difference Norm: ',avg_grad_norm_diff.item(),', Avg Surr Grad Norm: ',avg_surr_grad_norm.item(),', Avg True Grad Norm: ',avg_true_grad_norm.item())
                
            return avg_grad_norm_diff.item() #grad_loss.item()
        
        avg_norm_diff[1] = compare_gradients(self.grad_dataloader_center, model, get_grad_surrogate_point)
        
        
        def compare_hessians(surrogate_model):
            
            for X_hess, y_hess in self.hess_dataloader: 
                
                avg_hess_norm_diff = torch.mean(torch.norm(y_hess - self.get_hess_surrogate_point(surrogate_model, X_hess), p=2, dim=-1)/torch.max(torch.norm(y_hess,p=2,dim=-1),torch.norm(self.get_hess_surrogate_point(surrogate_model, X_hess), p=2, dim=-1)))  # torch.Tensor([0.01])))                
                avg_true_hess_norm = torch.mean(torch.norm(y_hess, p=2, dim=-1))
                avg_surr_hess_norm = torch.mean(torch.norm(self.get_hess_surrogate_point(surrogate_model, X_hess), p=2, dim=-1))
                print('Avg Hess Difference Norm: ',avg_hess_norm_diff.item(),', Avg Surr Hess Norm: ',avg_surr_hess_norm.item(),', Avg True Hess Norm: ',avg_true_hess_norm.item())

            return avg_hess_norm_diff.item() #grad_loss.item()

        if self.COMPUTE_HESS:            
            avg_norm_diff[2] = compare_hessians(model)  


        return loss_values, loss_test_values, loss_values_epoch, loss_test_values_epoch, err_values, err_test_values, err_values_epoch, err_test_values_epoch, err_extra_values_epoch, avg_norm_diff, model
    



    def run_exp(self, function_set= Experiment.FUNCTION_SET1, dim=20):

        #Define the datastructures used to make the histograms
        self.table_latex_dict_lstsq[0] = {}          
        self.table_latex_dict_lstsq[1] = {} 
        self.table_latex_dict_lstsq[2] = {} 


        self.table_latex_dict_NN[0] = {}          
        self.table_latex_dict_NN[1] = {} 
        self.table_latex_dict_NN[2] = {} 

        
        # Run the first set of fuctions for which dimensions need to be changed    
        for idx in function_set:

            self.table_latex_dict_lstsq[0][idx] = {}          
            self.table_latex_dict_lstsq[1][idx] = {} 
            self.table_latex_dict_lstsq[2][idx] = {} 

            self.table_latex_dict_NN[0][idx] = {}          
            self.table_latex_dict_NN[1][idx] = {} 
            self.table_latex_dict_NN[2][idx] = {}

            function = function_set[idx]
            train_dataloader, test_dataloader, _, grad_dataloader, inputsize, center, delta = self.create_problem(function, self.FDToll_NN1, dim)

            for op in self.lstsq_options:             
                
                surrogate_model =  NN_lstsq(inputSize = inputsize,option=op)

                surrogate_model.center = center
                surrogate_model.delta  = delta

                _, avg_norm_diff, surrogate_model = self.lstsq_opt(surrogate_model, train_dataloader, grad_dataloader, self.get_grad_surrogate_point)

                self.table_latex_dict_lstsq[0][idx][op] = avg_norm_diff[0] # function values difference
                self.table_latex_dict_lstsq[1][idx][op] = avg_norm_diff[1] # gradient difference
                self.table_latex_dict_lstsq[2][idx][op] = avg_norm_diff[2] # Hessian difference

            op = 12
            surrogate_model = RBF_interp(inputsize, rbf_gaussian, num_kernels=len(train_dataloader.dataset))
            surrogate_model.center = center
            surrogate_model.delta  = delta

            avg_norm_diff, surrogate_model = self.rbf_opt(surrogate_model, train_dataloader, grad_dataloader, self.get_grad_surrogate_point)

            self.table_latex_dict_lstsq[0][idx][op] = avg_norm_diff[0] # function values difference
            self.table_latex_dict_lstsq[1][idx][op] = avg_norm_diff[1] # gradient difference
            self.table_latex_dict_lstsq[2][idx][op] = avg_norm_diff[2] # Hessian difference
            for af in self.afunc:
                numNeurons_val = 4* inputsize

                #Create neural network
                model_NN = NN(inputSize = inputsize, outputSize = 1, numLayers = self.NUM_LAYERS, activationFunct = self.afunc[af], numNeurons = numNeurons_val)
                model_NN.center = center
                model_NN.delta  = delta
                #Train model on number on train_dataloader

                _, _, _, _, _, _, _, _, _, avg_norm_diff, model_NN = self.SGD_opt_with_grad(model_NN, 0.01, train_dataloader, test_dataloader, grad_dataloader, self.get_grad_surrogate_point, torch.tensor([])) 
                self.table_latex_dict_NN[0][idx][af] = avg_norm_diff[0] # function values difference
                self.table_latex_dict_NN[1][idx][af] = avg_norm_diff[1] # gradient difference
                self.table_latex_dict_NN[2][idx][af] = avg_norm_diff[2] # Hessian difference


    

if __name__ == "__main__":

    exp = Experiment_sect3()
    string_pickle_lstsq = os.getcwd() + '/data/table_latex_dict_set1_lstsq'
    string_pickle_NN    = os.getcwd() + '/data/table_latex_dict_set1_NN'

    exp.run_exp(exp.FUNCTION_SET1)

    with open(string_pickle_lstsq, 'wb') as handle:
            pickle.dump(exp.table_latex_dict_lstsq, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    with open(string_pickle_NN, 'rb') as handle:
            pickle.dump(exp.table_latex_dict_NN, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    exp.draw_histogram(string_pickle_lstsq, 1)

    exp.draw_histogram(string_pickle_NN, 0)
    


    exp2 = Experiment_sect3()


    string_pickle_lstsq = os.getcwd() + '/data/table_latex_dict_set2_lstsq'
    string_pickle_NN2   = os.getcwd() + '/data/table_latex_dict_set2_NN'

    # if not os.path.isfile(string_pickle_lstsq):

    exp2.run_exp(exp2.FUNCTION_SET2)

    with open(string_pickle_lstsq, 'wb') as handle:
            pickle.dump(exp2.table_latex_dict_lstsq, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    with open(string_pickle_NN2, 'wb') as handle:
            pickle.dump(exp2.table_latex_dict_NN, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    exp2.draw_histogram(string_pickle_lstsq, 1)
    exp2.draw_histogram(string_pickle_NN2, 0)






    
