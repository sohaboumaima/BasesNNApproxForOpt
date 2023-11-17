import math
import torch
import functorch 
import numpy as np
import sympy as sym
from nn_model import NN, NN_lstsq, RBF_interp #, anotherNN
from utils import DataPoints, rbf_gaussian
from torch.utils.data import DataLoader, random_split
import time
from torch.optim import LBFGS
import pycutest
import random


torch.set_default_dtype(torch.float64)

class FullLowEvalOpt:

    def __init__(self, x0, fun, cutest_prob, init_step, option=0, FDtype=0, alpha_0=1, gamma=2, theta=0.5, tol_alpha=10**-10):
        # option: 0= FLE, 1=FLE-S Natural Basis, 2=FLE-S Sigmoid Basis, 3=FLE-S RBFs, 4=FLE-S NN ReLU, 5=FLE-S NN SiLU 

        self.x0 = x0
        self.FDtype       = FDtype 
        self.current_eval = init_step #"FE"
        self.alpha        = alpha_0
        self.gamma        = gamma
        self.theta        = theta
        self.tol_alpha    = tol_alpha
        self.x            = x0
        self.x_prev       = x0
        self.iterates     = [self.x.clone().detach()]
        self.func_eval    = 0
        self.n            = self.x.size(dim=0)
        self.cutest_prob  = cutest_prob
        self.rbfs         = False
        
        self.set_option_param(option)
        def fun_aux(x):        
          if x.requires_grad:
            x = torch.squeeze(x).detach().numpy()  
          else:
            x = torch.squeeze(x).numpy()         
          out = self.cutest_prob.obj(x,gradient=False)
          out = torch.Tensor([out])
          return out
        self.fun_name     = fun
        self.fun          = fun_aux #self.cutest_prob.obj(x, gradient=False)#fun
        self.f            = self.fun(self.x.clone().detach())
        self.nfailed      = 0
        self.H            = torch.eye(self.n, dtype = torch.float64)
        self.points_grad  = []
        self.gradients    = []
        self.points       = []
        self.values       = []
        self.grad_simplex = False 
        self.g            = self.finitedifference()
        self.gFD          = self.g.detach().clone()
        self.evalmax      = 1000*self.n
        self.fesucsteps   = 0
        self.feunssteps   = 0
        self.lesucsteps   = 0
        self.leunssteps   = 0
        self.func_eval    += 1
        self.fvalhist     = [self.f]
        self.fevalhist    = [self.func_eval] 
        self.surrogate_model = None
        self.usesurrogate = False # This is a flag used to build the surrogate model when enough points have been collected and it is only used if self.surr is True
        self.failure      = 0 
        self.search_step_flag = False
        self.initial_search_step = False
        self.nosurr_after_initial_search_step = True # Only if self.initial_search_step == True
        self.direct_search_2dir = True #If True, in LE iterations, explore a randomly generated direction and its opposite; Otherwise, coordinate directions are used
        
        if (self.surr):
            self.points = [self.x.clone().detach()]
            self.values = [self.f]
            self.FDtoll_exp_surrogate = 8 #2 # exponent for FD tollerance used when using the surrogate model
        self.backtrack = 0

    
    def set_option_param(self, op):
        

        if (op > 5 or op < 0):
            raise Exception("option not recognized, please choose a number between 0 and 5")
        if (op == 0):
            self.surr = False
        else:
            self.max_num_points = (self.n+1)*(self.n+2)/2 + self.n + 1
            self.surr = True
            if (op == 1 or op ==2):
                self.lstsq = True
                self.min_num_points = (self.n+1)*(self.n+2)/2
                if (op==1):
                    self.regress_option = 1
                else:
                    self.regress_option = 8
            elif op == 3:
                self.lstsq = True
                self.rbfs = True
                self.min_num_points = (self.n+1)*(self.n+2)/2
                
            else:
                self.lstsq = False
                self.min_num_points = 0.2*(self.n+1)*(self.n+2)/2
                if op==4:
                    self.activation_function = torch.nn.SiLU()
                else:
                    self.activation_function = torch.nn.ReLU()
                self.num_epochs = 5
            


        

    def finitedifference(self):
        h = math.sqrt(np.finfo(float).eps)
        I = torch.eye(self.n)
        g = torch.zeros_like(self.x)
        
        x_temp = {}
        
        if self.surr and self.grad_simplex:
            h = 1e-8
            g_simplex = {}
            for i in range(self.n+1):
                g_simplex[i] = torch.zeros_like(self.x)
            A_simplex = [] # contains points y-y0
            A_simplex.append(torch.transpose(self.x,0,1))
            b_simplex = [] # contains f(y)
            b_simplex.append(self.f)

        for i in range(self.n):

            if self.FDtype==0:
                x_temp[i] = self.x + h * I[:,i:i+1]
                fh = self.fun(x_temp[i])
                
                g[i] = (fh - self.f)/h
                self.func_eval += 1
            
            if self.FDtype==1:
                fh1 = self.fun(self.x + h * I[:,i:i+1])
    
                fh2 = self.fun(self.x - h * I[:,i:i+1])                
                
                g[i] = (fh1 - fh2)/(2*h) 
                self.func_eval += 2
            
            if self.surr:
                if self.FDtype==1:
                    self.points.append((self.x + h * I[:,i:i+1]).clone().detach())
                    self.values.append(fh1)
                    self.points.append((self.x - h * I[:,i:i+1]).clone().detach())
                    self.values.append(fh2)
                if self.FDtype==0:    
                    self.points.append((self.x + h * I[:,i:i+1]).clone().detach())
                    self.values.append(fh)
                
            
            if self.surr and self.grad_simplex: 
                A_simplex.append(torch.transpose(x_temp[i],0,1))
                b_simplex.append(torch.Tensor([fh]))  
                
        if self.surr and self.grad_simplex:
            A_simplex = torch.cat(A_simplex)
            b_simplex = torch.cat(b_simplex).reshape(-1,1)
            for i in range(self.n+1):
                A_temp = A_simplex - A_simplex[i]
                A = A_temp[[j for j in range(A_temp.shape[0]) if j != i], :]
                b_temp = b_simplex - b_simplex[i]
                b = b_temp[[j for j in range(b_temp.shape[0]) if j != i], :]
                g_simplex[i] = torch.linalg.lstsq(A, b).solution
            
        if self.surr:
            self.points_grad.append(self.x.clone().detach())
            self.gradients.append(g) 
            
            self.gFD = g.detach().clone()
            
            if self.grad_simplex:
                for i in range(self.n):
                    self.points_grad.append(x_temp[i].clone().detach())
                    self.gradients.append(g_simplex[i+1].reshape(-1,1))  # we skip i==0, which is associated with self.x, because the FD gradient is more accurate than the simplex gradient
        return g


    def remove_close_points(self, points, values):
        """
        This function removes points that are exactly the same.
        points is like [torch.Tensor([[1.0000],[1.0000],[1.0000]]), torch.Tensor([[1.0000],[1.0000],[1.0000]]), torch.Tensor([[2.0000],[1.0000],[-1.0000]])]
        values is like [torch.Tensor([25.0000]),torch.Tensor([25.0000]),torch.Tensor([26.0000])]
        """
        aa_cat = torch.cat(points, dim=1)
        values = [e.reshape(1,1) for e in values]
        bb_cat = torch.cat(values).reshape(1,-1)
        cc_cat = torch.cat((aa_cat,bb_cat),dim=0)
        cc_uniq, indices = torch.unique(cc_cat, dim=1, return_inverse=True)
        points_uniq = [cc_uniq[:-1, i].reshape(-1,1) for i in range(cc_uniq.shape[1])]
        values_uniq = [cc_uniq[-1:, i].reshape(-1,1) for i in range(cc_uniq.shape[1])]
        return points_uniq, values_uniq
    
    
    def add_points_values(self, point_to_add, value_to_add):
        "Check if tensor_to_add is already in points and, if not, add it to points"
        tensor_in_list = False
        for tensor in self.points:
            if torch.all(torch.eq(tensor, point_to_add)):
                tensor_in_list = True
                break
        if not tensor_in_list:
            self.points.append(point_to_add)
            self.values.append(value_to_add)
    

    def init_surrogate_model(self):

        self.times_trained = 0
        
        # Create NN
        if self.lstsq == False:

            self.surrogate_model =  NN(inputSize = self.n, outputSize = 1, numLayers = 3, activationFunct = self.activation_function, numNeurons = 4*self.n)
            
        else:
            if self.rbfs:
                self.surrogate_model = RBF_interp(self.n, rbf_gaussian)
            else:
                self.surrogate_model =  NN_lstsq(inputSize = self.n, option=self.regress_option)

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

    def train_NN(self, model, learning_rate, train_dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("---------------------------------------------------------")

        for epoch in range(self.num_epochs):
            it = 0
            for X, y in train_dataloader:
                
                loss = self.loss_function(X, y, model)
                loss_print = loss.item()
                
                    
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Compute the gradient
                loss.backward()
                optimizer.step()
                
                it = it + 1
                                                
        print("Training Complete")
        print("Total number of trainable parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("---------------------------------------------------------")


    def train_surrogate_model(self):

        learning_rate = 0.01 

        x_new = self.x + torch.Tensor(1, self.n).uniform_(-0.1, 0.1).T
        self.points.append(x_new)
        self.values.append(self.fun(x_new.clone().detach()))
       
        self.func_eval += 1

        try:
            val = torch.cat(self.values)
        except:
            val = torch.tensor(self.values)

        if not self.lstsq:  
            
            # Dataset for neural network
            dataset = DataPoints(torch.transpose(torch.cat(self.points, 1), 0, 1), val)

            train_set_size = int(len(dataset))
            train_set, test_set = random_split(dataset, [train_set_size, 0]) 
            
           
            train_dataloader = DataLoader(dataset=train_set, batch_size=math.ceil(0.1*train_set_size), shuffle=True)

            if self.times_trained == 0:

                center = torch.tensor(self.x0, dtype=torch.float64).reshape(1,-1)

                delta = 0

                for X_batch,_ in train_dataloader:
                    batch_max_norm = torch.norm(X_batch, p=2, dim=1).max()
                    delta = max(delta, batch_max_norm.item())
                
                self.surrogate_model.center = center
                self.surrogate_model.delta  = delta

            else:
                self.num_epochs = 1                


                self.train_NN(self.surrogate_model, learning_rate, train_dataloader)

            
       
        else:
            
            # Dataset for function interpolation
            Y = torch.cat(self.points, 1)
            f_values = val.clone()
            Ynorms = Y - self.x
            Ynorms, indices = torch.sort(torch.norm(Ynorms, p=2, dim=0))
                
            Y = Y[:,indices]
            f_values = f_values[indices]
            c1 = torch.Tensor([0.1])
            indices2 = Ynorms <= c1
            count_close_points = torch.sum(indices2.int())
            while True: 
                if count_close_points < self.min_num_points:
                    c1 = 1.1*c1

                else:
                    break
                print('HERE---------: ',c1.item(),', dim Y = ',count_close_points.item(),', dim Ynorms',Ynorms.shape[0])
                indices2 = Ynorms <= c1
                count_close_points = torch.sum(indices2.int())  

                          
            Y = Y[:,indices2] 
            f_values = f_values[indices2]

        
            print('dim Y = ',Y.shape,', Threshold = ',c1.item(),', norm g = ',torch.norm(self.g).item())

            shifted_X = Y - self.x
            row_norms = torch.norm(shifted_X, dim=1)
            max_norm = torch.max(row_norms) 
            self.surrogate_model.center = torch.transpose(self.x,0,1)
            self.surrogate_model.delta = max_norm
    
            dataset = DataPoints(torch.transpose(Y, 0, 1), f_values)
            
            # Random split of the dataset
            train_set_size = int(len(dataset))
            
            train_set, _ = random_split(dataset, [train_set_size, 0]) #test_set_size
            
            train_dataloader = DataLoader(dataset=train_set, batch_size=train_set_size, shuffle=True) #batch_size=4 #batch_size=train_set_size

            if self.rbfs:
                self.surrogate_model.interpolate(train_dataloader)
            else:
                self.lstsq_opt(self.surrogate_model, train_dataloader)
        
        self.times_trained += 1
        
            
    
    def lstsq_opt(self, surrogate_model, train_dataloader):
        A_lstsq = []
        b_lstsq = []
        for X, y in train_dataloader:
            ## shift and scale data June 2023
            shifted_X = X - torch.transpose(self.x,0,1)
            row_norms = torch.norm(shifted_X, dim=1)
            max_norm = torch.max(row_norms) 
            surrogate_model.center = torch.transpose(self.x,0,1)
            surrogate_model.delta = max_norm
            A_lstsq.append(torch.cat((torch.ones(X.shape[0],1),surrogate_model.compute_activations(X)),dim=1))
            b_lstsq.append(y)
        A_lstsq = torch.cat(A_lstsq)
        b_lstsq = torch.cat(b_lstsq).reshape(-1,1)
        
            
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
    
    
    def get_grad_surrogate(self):
        
        # self.surrogate_model.eval() #AnotherNN
        y = torch.transpose(self.x, 0, 1) 
        y.requires_grad_() 
        pred = self.surrogate_model(y)
        pred.backward(retain_graph=True)
        g = y.grad.clone().detach()
        y.grad.zero_()
        return torch.transpose(g, 0, 1)


    def get_hessian_surrogate(self):
        
        y = torch.transpose(self.x, 0, 1) 
        y.requires_grad_() 
        
        # Compute first-order gradients
        pred = self.surrogate_model(y)
    
        grad = torch.autograd.grad(pred, y, create_graph=True)[0]
        
        # Compute second-order gradients and construct Hessian matrix
        H = torch.zeros((len(self.x), len(self.x)))
        for i in range(len(self.x)):
            H[i,:] = torch.autograd.grad(grad[0,i], y, create_graph=True)[0].detach()
        
        return H 
    
    
    def get_grad_surrogate_finitedifference(self):
    
        h = 10**-self.FDtoll_exp_surrogate
        I = torch.eye(self.n)
        g = torch.zeros_like(self.x)

        with torch.no_grad():
                
            for i in range(self.n):
    
                fh1 = self.surrogate_model(torch.transpose(self.x + h * I[:,i:i+1], 0, 1))

                fh2 = self.surrogate_model(torch.transpose(self.x - h * I[:,i:i+1], 0, 1))                
                
                g[i] = (fh1 - fh2)/(2*h)            
            
        return g
    

    def get_grad_surrogate_point(self, model, xx):
        'Compute the gradient of the surrogate model at each of the points in the batch xx'
        
        g_rows = []
        for i in range(xx.shape[0]):    
            y = torch.Tensor(xx[i]).reshape(1,-1)
            y.requires_grad_()
            pred = model(y)
            g_rows.append(torch.autograd.grad(pred, y, create_graph=True)[0])
        g = torch.stack(g_rows)
        
        return torch.transpose(g, 0, 1)
    
    
    def get_grad_obj_function(self):
        y = self.x.clone().detach()
        y.requires_grad_()
        pred = self.fun(y)
        pred.backward(retain_graph=True)
        g = y.grad.clone().detach()
        y.grad.zero_()
        return torch.transpose(g, 0, 1)


    def get_grad_obj_function_cutest(self):
          if self.x.requires_grad:
            x = torch.squeeze(self.x).detach().numpy()  
          else:
            x = torch.squeeze(self.x).numpy()         
          _, g = self.cutest_prob.obj(x,gradient=True)
          g = torch.Tensor(g)
          return g.reshape(1,-1)
          
    def is_sufficent_decrease(self, ftemp, rho):

        if (ftemp <= self.f - rho):
            return True
        else:
            return False
            

    def Low_eval_step(self):
        v = torch.randn(self.n,1, dtype = torch.float64)
        rho = min(1e-5, 1e-5 *self.alpha**2)

                
        if self.surr and self.usesurrogate and self.search_step_flag:# and norm_g > 1e-1: 
                xout = self.search_step()
                xcurr = xout
        else:
            xcurr = self.x

        if self.direct_search_2dir:             
          xtemp = xcurr + self.alpha * v
          ftemp = self.fun(xtemp)
  
          if self.surr:
              self.points.append(xtemp.clone().detach())
              self.values.append(ftemp)
  
          success = False
          self.func_eval += 1
          if self.is_sufficent_decrease(ftemp, rho):
              self.f = ftemp
              self.x = xtemp
              success = True
          
          else:
              xtemp = xcurr - self.alpha * v
              ftemp = self.fun(xtemp)
              if self.surr:
                  self.points.append(xtemp.clone().detach())
                  self.values.append(ftemp)
                 
                 
              self.func_eval += 1
              if self.is_sufficent_decrease(ftemp, rho):
                  self.f = ftemp
                  self.x = xtemp
                  success = True

        else:
            I = torch.eye(self.n)

            for i in range(self.n):
                
                xtemp = self.x + self.alpha * I[:,i:i+1]
                ftemp = self.fun(xtemp)
    
                if self.surr:
                    self.points.append(xtemp.clone().detach())
                    self.values.append(ftemp)
                    
                    
                success = False
                self.func_eval += 1
                if self.is_sufficent_decrease(ftemp, rho):
                    self.f = ftemp
                    self.x = xtemp
                    success = True
                    break
                
                xtemp = self.x - self.alpha * I[:,i:i+1]
                ftemp = self.fun(xtemp)
    
                if self.surr:
                    self.points.append(xtemp.clone().detach())
                    self.values.append(ftemp)
                
    
                success = False
                self.func_eval += 1
                if self.is_sufficent_decrease(ftemp, rho):
                    self.f = ftemp
                    self.x = xtemp
                    success = True
                    break
                    

        self.fvalhist.append(self.f.item())
        self.fevalhist.append(self.func_eval)

        if success:
            self.alpha *= self.gamma
            self.lesucsteps +=1
        else:
            self.alpha *= self.theta
            self.nfailed += 1
            self.leunssteps += 1

        if (self.nfailed >= self.backtrack):
            self.current_eval = "FE"
            self.nfailed = 0         


    def search_step(self):
       
        
        lb = torch.transpose(self.x, 0, 1)-0.1*torch.ones(self.n).reshape(1,-1)
        ub = torch.transpose(self.x, 0, 1)+0.1*torch.ones(self.n).reshape(1,-1)
        
        
        # Define the projection operation to enforce the bound constraints
        def project_ellinfty(x, lb, ub):
            return torch.min(torch.max(x, lb), ub)

        # Define the projection operation to enforce the bound constraints
        def project_ell2(x, radius):
            x *= radius / torch.max(radius,torch.norm(x))
            return x
        
        # Define the closure function for LBFGS optimizer
        def closure():
            optimizer.zero_grad()
            # loss = quadratic_obj(xtemp, H, g)
            loss = self.surrogate_model(xtemp)
            loss.backward()
            return loss
        
        # Initialize the solution variable
        xtemp = torch.transpose(self.x, 0, 1).clone().detach() #self.x.clone().detach()
        xtemp.requires_grad_()
        
        print('pre ',xtemp)
        
        # Define the optimizer
        optimizer = LBFGS([xtemp], lr=0.1, max_iter=1)        
        # optimizer = LBFGS([xtemp], lr=0.1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn="strong_wolfe")
        
        print('f ',self.surrogate_model(xtemp))
        
        # Optimize the objective function subject to the bound constraints
        for i in range(5):
            optimizer.step(closure)
            xtemp.data = project_ellinfty(xtemp.data, lb, ub)
            # xtemp.data = project_ell2(xtemp.data, radius=torch.Tensor([1]))
            print('f ',self.surrogate_model(xtemp),', true f',self.fun(xtemp.reshape(-1,1)))
        
        print('after ',xtemp)
        
        xout = xtemp.reshape(-1,1)
        
        return xout.detach().clone()


    def armijo(self, d):
        beta = 1
        tau  = 0.5
        eta  = 1e-8
        rho = min(1e-5, 1e-3*self.alpha**2)
        

        while beta > 1e-16:
            xtemp = self.x + beta*d

            ftemp = self.fun(xtemp)
            self.backtrack += 1
            self.func_eval += 1

            if self.surr and (self.backtrack==1 or not self.lstsq): #june2023
                self.points.append(xtemp.clone().detach())
                self.values.append(ftemp)


            gd  = torch.dot(self.g.squeeze(), d.squeeze())

            if self.surr and self.usesurrogate: 
                if self.is_sufficent_decrease(ftemp, -eta*beta*gd.item()):

                    if ftemp < self.f:                                 
                        return beta, ftemp, False
                    else:

                        return 0, self.f, True
                
                elif (beta < rho):
                        return 0, self.f, True
    
                else:
                    beta *= tau
                    
            else:
                if self.is_sufficent_decrease(ftemp, -eta*beta*gd.item()):
                    return beta, ftemp, False
                
                elif (beta < rho):
                        return 0, self.f, True

                else:
                    beta *= tau

        return 0, self.f, True


    def Full_eval_step(self):

        self.xfe = self.x.detach().clone()
        self.gfe = self.g.detach().clone()

        self.backtrack = 0
        d  = torch.matmul(-self.H, self.g) 
        beta, ftemp, failed = self.armijo(d)
        print('Armijo ',self.func_eval)
        
        if not failed:
            self.f = ftemp
            self.x += beta * d
            self.fesucsteps += 1
        else:

            self.current_eval = "LE"
            self.feunssteps +=1

        self.fvalhist.append(self.f.item())
        self.fevalhist.append(self.func_eval)
        return failed


    def update_BFGS_matrix(self):

        s = self.x - self.xfe
        y = self.g - self.gfe

        c = torch.dot(s.squeeze(), y.squeeze()).item()
        
        I = torch.eye(self.n, dtype = torch.float64)

        if (c > 1e-10 * torch.norm(s).item()*torch.norm(y).item()):
            r = 1/c
            matmul = torch.matmul(I-r* torch.matmul(s, y.T), self.H) 
            self.H = torch.matmul(matmul,I-r*torch.matmul(y, s.T)) + r*torch.matmul(s, s.T)
            
    
    def update_Newton_matrix(self):

       self.H = torch.linalg.inv(self.get_hessilstsqan_surrogate())


    def Full_Low_Eval(self):

        model_init = False
        best_test_error = torch.inf
        
        it = 0
        
        count_same_val = 0 # to count how many times the objective function value does not significantly decrease
        prev_f = 0
        count_iter_since_last_train = 0
        
        print('***** FLE - {10} - Iter: {0} - {1} - F: {2:1.3g} - Feval: {3} - FES: {4} - FEU: {5} - LES: {6} - LEU: {7} - nfailed: {8} - backtrack: 0 - alpha: {9:1.3g} - {11}'.format(it,self.current_eval,self.f.item(),self.func_eval,self.fesucsteps,self.feunssteps,self.lesucsteps,self.leunssteps,self.nfailed,self.alpha,self.fun_name,self.current_eval))               

        while ((self.func_eval < self.evalmax and self.alpha > self.tol_alpha) and count_same_val <= 10*10):            

            self.iter = it 
            init_current_eval = self.current_eval
            

            if (self.surr and len(self.values) >= self.min_num_points and not model_init):
                
                self.first_iter_surr = it
                
                self.usesurrogate = True
                self.init_surrogate_model()

                # Generate points according to a uniform distribution and then minimize the model to find the best initial point
                if self.initial_search_step:
                    self.train_local_model()
                    xtemp = self.search_step()
                    print('AA',self.f)
                    ftemp = self.fun(xtemp)
                    self.f = ftemp
                    print('BB',self.f)
                    self.x = xtemp
                    self.func_eval += 1
                    self.search_step()
                    if self.nosurr_after_initial_search_step:
                        self.usesurrogate = False

                model_init = True                

            if (self.current_eval == "FE"):
                self.Full_eval_step()

            else:

                if self.surr and self.usesurrogate and self.search_step_flag: 
                    test_error = self.train_local_model()
                self.Low_eval_step()
            
           


            if self.current_eval=="FE":
                if self.surr and self.usesurrogate and not self.search_step_flag:
                    self.train_surrogate_model() 
                    self.g = self.get_grad_surrogate()
                    count_iter_since_last_train = 0 
                        
                else:
                    count_iter_since_last_train += 1
                    # we do not want to compute the gFD again if we have not updated the current iterate
                    # if not torch.all(torch.eq(self.x,self.x_prev)):
                    self.g = self.finitedifference()

                    print('gFD    {0:1.3g} {1:1.3g}, norm: {2:1.3g}'.format(self.g[0][0].item(),self.g[1][0].item(),torch.norm(self.g)))
                    self.x_prev = self.x.detach().clone()

                self.update_BFGS_matrix()                
                
            
            self.iterates.append(self.x.clone().detach())
            
            
            if prev_f - self.f.item() >= 1e-16 and prev_f - self.f.item() <= 1e-5 and model_init: #and not self.usesurrogate:
                count_same_val += 1
                
            prev_f = self.f.item()
            
            # print('f',self.f.item())
            
            final_current_eval = self.current_eval
                        
            it += 1
            
            print('***** FLE - {11} - Iter: {0} - {1} - F: {2:1.10g} - Feval: {3} - FES: {4} - FEU: {5} - LES: {6} - LEU: {7} - nfailed: {8} - backtrack: {9} - alpha: {10:1.3g} - from {12} to {13}'.format(it,self.current_eval,self.f.item(),self.func_eval,self.fesucsteps,self.feunssteps,self.lesucsteps,self.leunssteps,self.nfailed,self.backtrack,self.alpha,self.fun_name,init_current_eval,final_current_eval))               
            






        

        
        

