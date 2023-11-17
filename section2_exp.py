import torch
from  torch.nn import *
from nn_model import NN
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiment import Experiment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_default_dtype(torch.float64)

## This file runs the experiments in section 2 where we investigate the choice of activation functions on function approximation
## We use a Feedforward model with 2 hidden layers and as test functions, we use two groups of functions from the Cutest Library
## One for which we increase the dimension, and a fixed set defined below 



class Experiment_sect2(Experiment):

    NUM_LAYERS = 3


    def __init__(self, activ_functions={'NN ReLU':torch.nn.ReLU(), 'NN ELU': torch.nn.ELU(), 'NN SiLU': torch.nn.SiLU(), 'NN Sigmoid': torch.sigmoid, 'NN Tanh':torch.tanh}, epochs=300, lr=1e-2, plot=False):

        super().__init__(activ_functions, plot)

        self.epochs = epochs
        self.learning_rate = lr


    def train(self, model, train_dataloader, test_dataloader, dataloader_extra):
        
        loss_values = []
        loss_test_values = []
    
        loss_values_epoch = []
        loss_test_values_epoch = []
    
        err_values = []
        err_test_values = []
        
        err_values_epoch = []
        err_test_values_epoch = []
        err_extra_values_epoch = []
                
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)
        
        # Print values of the parameters
        g = 0
        for param in model.parameters():
            if g == len(list(model.parameters()))-1:
                print('Weight OL: ',int(g/2),':',param.data)
            elif g % 2 == 0:
                print('Weight HL ',int(g/2),':',param.data)
            elif g % 2 == 1:
                print('Bias: ',int(g/2),':',param.data)
            g = g + 1

        print("---------------------------------------------------------")
        
        start_time = time.time()
        
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

                # Compute the error on the extra dataset
                err_extra_tot = 0
                it = 0
                for X_extra, y_extra in dataloader_extra:
                    err_extra = self.error_function(X_extra, y_extra, model)
                    err_extra_tot += err_extra.item()
                    it += 1
                    
                err_extra_values_epoch.append(err_extra_tot/it)
            model.train()

            # Update the learning rate scheduler based on validation loss
            scheduler.step(validation_loss)
            running_time = time.time() - start_time
            print('------ Epoch: ',epoch,' - Iter: ',it,' - Training loss: ',loss_print,' - Testing loss: ',loss_test_values[len(loss_test_values)-1],' - Error: ',err.item(),' - Testing Error: ',err_test_tot,' - Time: ',running_time)   
               
        print("Training Complete")
        print("Total number of trainable parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("---------------------------------------------------------")
        
        return loss_values, loss_test_values, loss_values_epoch, loss_test_values_epoch, err_values, err_test_values, err_values_epoch, err_test_values_epoch, err_extra_values_epoch
    
    
    def run_exp(self):

        #Define the datastructures used to make the performance profiles
        # objective_fvals_dict = {}
        # objective_fvals_train_dict = {}
        objective_fvals_loss_dict = {}
        objective_fvals_loss_train_dict = {}

        # Loop over the activation functions
        for dim in [20, 40, 60]:# self.dimensions:
        

            # objective_fvals_dict[dim] = {}
            # objective_fvals_train_dict[dim] = {}
            objective_fvals_loss_dict[dim] = {}
            objective_fvals_loss_train_dict[dim] = {}
            # Run the first set of fuctions for which dimensions need to be changed
            
            for af in self.afunc:
                # objective_fvals_dict[dim][af] = {}
                # objective_fvals_train_dict[dim][af] = {}
                objective_fvals_loss_dict[dim][af] = {}
                objective_fvals_loss_train_dict[dim][af] = {}
                 
                for idx in self.FUNCTION_SET1:

                    #Create problem from dimension 
                    function = self.FUNCTION_SET1[idx]
                    train_dataloader, test_dataloader, dataloader_extra, grad_dataloader, inputsize, center, delta = self.create_problem(function, self.FDToll_NN1, dim)
                    numNeurons_val = 4* inputsize

                    #Create neural network
                    model_NN = NN(inputSize = inputsize, outputSize = 1, numLayers = self.NUM_LAYERS, activationFunct = self.afunc[af], numNeurons = numNeurons_val)

                    model_NN.center = center
                    model_NN.delta  = delta
                    #Train model on number on train_dataloader

                    loss_values_NN, loss_test_values_NN, loss_values_epoch_NN, loss_test_values_epoch_NN, err_values_NN, err_test_values_NN, err_values_epoch_NN, err_test_values_epoch_NN, err_extra_values_epoch_NN = self.train(model_NN, train_dataloader, test_dataloader, dataloader_extra)

                    # objective_fvals_dict[dim][af][idx] = np.array(err_test_values_epoch_NN)
                    # objective_fvals_train_dict[dim][af][idx] =  np.array(err_values_epoch_NN)
                    objective_fvals_loss_dict[dim][af][idx] = np.array(loss_test_values_epoch_NN)
                    objective_fvals_loss_train_dict[dim][af][idx] = np.array(loss_values_epoch_NN)


                    if self.plot:
                        string_label = 'Train Error NN ' + af + '-'+ str(dim)    
                        plt.plot(err_values_epoch_NN, label=string_label)
                        string_label = 'Test Error NN ' + af + '-'+ str(dim)  
                        plt.plot(err_test_values_epoch_NN, label=string_label)
                        plt.xlabel("Epochs", fontsize = 13)
                        plt.legend(frameon=False, loc="upper right")
                        string = function + ', Dim = ' + str(dim) 
                        plt.title(string)
                        if self.save_plot:
                            fig = plt.gcf()
                            string = os.getcwd() + '/by_functions/' + function + '_' + str(inputsize) + '_' + str(self.epochs)+ '_' + af + '.png'
                            fig.savefig(string, dpi=100)

                        plt.close()
            with open(os.getcwd() + '/data/section2_SiLUELUinit_dump_set1_dim' + str(dim), 'wb') as handle:
                pickle.dump([objective_fvals_loss_dict[dim], objective_fvals_loss_train_dict], handle, protocol=pickle.HIGHEST_PROTOCOL) 

        # objective_fvals_dict_2 = {}
        # objective_fvals_train_dict_2 = {}
        objective_fvals_loss_dict_2 = {}
        objective_fvals_loss_train_dict_2 = {}

        
        for af in self.afunc:
            # objective_fvals_dict_2[af] = {}
            # objective_fvals_train_dict_2[af] = {}
            objective_fvals_loss_dict_2[af] = {}
            objective_fvals_loss_train_dict_2[af] = {}

            for idx in self.FUNCTION_SET2:

                function = self.FUNCTION_SET2[idx]
                train_dataloader, test_dataloader, dataloader_extra, _, inputsize, center, delta = self.create_problem(function, self.FDToll_NN1, 4)
                numNeurons_val = 4* inputsize

                #Create neural network
                model_NN = NN(inputSize = inputsize, outputSize = 1, numLayers = self.NUM_LAYERS, activationFunct = self.afunc[af], numNeurons = numNeurons_val)
                model_NN.center = center
                model_NN.delta  = delta
                #Train model on number on train_dataloader

                loss_values_NN, loss_test_values_NN, loss_values_epoch_NN, loss_test_values_epoch_NN, err_values_NN, err_test_values_NN, err_values_epoch_NN, err_test_values_epoch_NN, err_extra_values_epoch_NN = self.train(model_NN, train_dataloader, test_dataloader, dataloader_extra)

                # objective_fvals_dict_2[af][idx] = np.array(err_test_values_epoch_NN)
                # objective_fvals_train_dict_2[af][idx] =  np.array(err_values_epoch_NN)
                objective_fvals_loss_dict_2[af][idx] = np.array(loss_test_values_epoch_NN)
                objective_fvals_loss_train_dict_2[af][idx] = np.array(loss_values_epoch_NN)


                if self.plot:
                    string_label = 'Train Error NN ' + af + '-'+ str(inputsize)    
                    plt.plot(err_values_epoch_NN, label=string_label)
                    string_label = 'Test Error NN ' + af + '-'+ str(inputsize)
                    plt.plot(err_test_values_epoch_NN, label=string_label)
                    plt.xlabel("Epochs", fontsize = 13)
                    # plt.gca().set_ylim([0,ylim_plot]) 
                    plt.legend(frameon=False, loc="upper right")
                    string = function + ', Dim = ' + str(inputsize) 
                    plt.title(string)
                    if self.save_plot:
                        fig = plt.gcf()
                        string =  os.getcwd() + '/by_functions/' + function + '_' + str(inputsize) + '_' + str(self.epochs)+ '_' + af + '.png'
                        fig.savefig(string, dpi=100)

                    plt.close()

        return objective_fvals_loss_dict, objective_fvals_loss_train_dict, objective_fvals_loss_dict_2, objective_fvals_loss_train_dict_2  #, objective_fvals_loss_dict_2, objective_fvals_loss_train_dict_2
    

        
if __name__ == "__main__":
    exp = Experiment_sect2()
    
    list_dicts = exp.run_exp()

    with open(os.getcwd() + '/data/section2_SiLUELUinit_dump_set2', 'rb') as handle:
        pickle.dump(list_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        # list_dicts = pickle.load(handle)


    save_perf_profiles = True
    namings = ['PPtest', 'PP']
    lns = {'NN ReLU': '-', 'NN ELU':'-.', 'NN SiLU': 'dashed', 'NN Sigmoid': '--', 'NN Tanh': 'dotted'}
    i = 0
    for dic in list_dicts[:2]:

        for dim in dic:
            # Tolerance 1e-2
            
            dataframe_1e2 = exp.draw_perf_profiles(dic, 1e-2, lns=lns, filename= namings[i] + "_dim" +str(dim) + "_m2_ReLUTanhSigmoid_loss_38probs.png")

            # Tolerance 1e-5
            dataframe_1e5 = exp.draw_perf_profiles(dic, 1e-5, lns=lns, filename= namings[i] + "_dim" +str(dim) + "_m5_ReLUTanhSigmoid_loss_38probs.png")

            if save_perf_profiles:
                string_dataframe_1e2 = os.getcwd() + '/data/' + namings[i] + '_set1_tau2.csv'
                dataframe_1e2.to_csv(string_dataframe_1e2, index='False')

                string_dataframe_1e5 = os.getcwd() + '/data/' + namings[i] + '_set1_tau5.csv'
                dataframe_1e5.to_csv(string_dataframe_1e5, index='False')

        i += 1

    i = 0
    for dic in list_dicts[2:]:
        # Tolerance 1e-2
        dataframe_1e2 = exp.draw_perf_profiles(dic, 1e-2, lns=lns, filename= namings[i] + "_dim00_m2_ReLUTanhSigmoid_loss_negcurvprobs.png")

        # Tolerance 1e-5
        dataframe_1e5 = exp.draw_perf_profiles(dic, 1e-5, lns=lns, filename= namings[i] + "_dim00_m5_ReLUTanhSigmoid_loss_negcurvprobs.png")

        if save_perf_profiles:
            string_dataframe_1e2 = os.getcwd() + '/data/' + namings[i] + '_set2_tau2.csv'
            dataframe_1e2.to_csv(string_dataframe_1e2, index='False')

            string_dataframe_1e5 = os.getcwd() + '/data/' + namings[i] + '_set2_tau5.csv'
            dataframe_1e5.to_csv(string_dataframe_1e5, index='False')

        i += 1
