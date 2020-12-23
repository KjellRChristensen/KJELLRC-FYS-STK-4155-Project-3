import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from TestPythonCode import KFold, KFoldBootstrap
    from multiprocessing import process
    from multiprocessing import freeze_support
    from numpy.lib.type_check import nan_to_num
    #from ThreadsStruct import Loss
    from math import inf, nan
    from os import X_OK, times, truncate
    from re import VERBOSE
    from autograd.differential_operators import deriv
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from sklearn.base import _UnstableArchMixin
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
    from sklearn import model_selection
    #from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn import datasets
    from sklearn.utils import resample
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import sys, os
    from random import random, seed
    from imageio import imread
    from numpy.random import normal, uniform
    from tqdm import tqdm
    from tqdm import tqdm_notebook, trange
    from ipywidgets import IntProgress
    from time import time
    from time import sleep
    from sklearn.datasets import load_breast_cancer
    from functools import lru_cache
    # from sklearn.metrics import accuracy_score, classification_report

    import os
    #os.nice(1)
    #os.nice()

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Plots", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)

theta_path_mgd=[]

#Global variables
CurrentPath ="Not assigned"
fig = plt.figure()
#ax = fig.gca(projection='3d')


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # Save plots to to ./Plots
    #
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Disable print(x) statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore 
def enablePrint():
    sys.stdout = sys.__stdout__

#blockPrint()
#enablePrint()

# Save plot figures in specific catalogue
def MakeDirAndReturnPath(Dir4Plots):
    #Create a new directory if needed and returns path.
    #directory = os.path.split(path)[0]
    cwd = os.getcwd()
    directory = cwd + Dir4Plots
    #filename = "%s.%s" % (os.path.split(path)[1], ext)
    #if directory == '':
    #    directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def SubMean(YPredict):
    YPredict = YPredict - np.mean(YPredict, axis=0)
    return YPredict

############ PROJECT 3 ###########
##################################
import threading
import time
#ThreadLock = threading.Lock()
#Threads = []
#TotalLoss = []
GlobalThread = threading.Condition()
#GlobalThread = threading.RLock()
#GlobalThread = threading.Lock()
flag = 1      #Start with Thread 1
val = 0

# Struct to keep status of all running threads and associated loss
GlobalEpoch=0
Threads=6
CleanUpDone=True
ThreadStatus= np.zeros(shape=(Threads,5))
for count in range(0,Threads):
    ThreadStatus[count,0]=count


class DeepNeuralNettworkMultiThreaded(threading.Thread):
    def __init__(self, Architecture, InitBias, InitWeight,ThreadID, Name, LockId,Flag,X_train,y_train,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end):
        threading.Thread.__init__(self)
        self.Architecture = Architecture
        self.Bias = InitBias
        self.InitWeight = InitWeight
        self.Params = self._initialize_params(Architecture) 
        self.ThreadID = ThreadID
        self.Name = Name
        self.LockId = LockId
        self.FlagId = Flag
        self.X_train=X_train
        self.y_train=y_train
        self.Epochs=Epochs
        self.LRate=LRate
        self.verbose=Verbose
        self.ShowLoss=ShowLoss
        self.SyncLoss=SyncLoss
        self.send_end=send_end
        self.ThreadNetId='ToBeChanged'
    
        
    def _initialize_params(self, architecture):
        params = {}
        for id_, layer in enumerate(architecture):
            layer_id = id_ + 1

            input_dim = layer['input_dim']
            output_dim = layer['output_dim']

            params['W'+str(layer_id)] = np.random.randn(output_dim, input_dim)*0.1
            params['b'+str(layer_id)] = np.zeros((output_dim, 1))

        return params
    
    def run(self):
        print ("Starting " + self.Name)
        # Get lock to synchronize threads
        #ThreadLock.acquire()
        self.fit(self.X_train, self.y_train, self.Epochs,self.LRate,self.verbose,self.ShowLoss)
        #self.print_time(self.Name, self.Counter, 3)
        # Free lock to release next thread
        Predict=self.predict(self, self.X_test,self.y_test)
        #ThreadLock.release()
        self.send_end.send(Predict)
        return

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    def relu(self, Z):
        return np.maximum(0, Z)
    #KRC - Cross Entropy
    def crossentropy(self, T, Y):
        E = 0
        for i in range(len(T)):
            if T[i] == 1:
                E -= np.log(Y[i])
            else:
                E -= np.log(1 - Y[i])
        return E

    def sigmoid_backward(self, dA, z_curr):
        sig = self.sigmoid(z_curr)
        return sig*(1-sig)*dA

    def relu_backward(self, dA, z_curr):
        dz = np.array(dA, copy=True)
        dz[z_curr<=0]=0
        return dz
  
    def _forward_prop_this_layer(self, A_prev, W_curr, b_curr, activation_function):
        z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation_function == 'relu':
            activation = self.relu
        elif activation_function == 'sigmoid':
            activation = self.sigmoid
        elif activation_function == 'crossentropy':
            activation = self.crossentropy
        else:
            raise Exception(f"{activation_function} is not supported, Only sigmoid, relu are supported")

        return activation(z_curr), z_curr

    def _forward(self, X):
        cache = {}
        A_current = X
        for layer_id_prev, layer in enumerate(self.Architecture):
            current_layer_id = layer_id_prev+1

            A_previous = A_current
            activation = layer['activation']

            W_curr = self.Params['W'+str(current_layer_id)]
            b_curr = self.Params['b'+str(current_layer_id)]

            A_current, Z_curr = self._forward_prop_this_layer(A_previous, W_curr,
                                                        b_curr, activation)

            cache['A'+str(layer_id_prev)] = A_previous
            cache['Z'+str(current_layer_id)] = Z_curr
            
        return A_current, cache

    def _criterion(self, y, yhat):
        m = yhat.shape[1]
        cost = -1/m * (np.dot(y, np.log(yhat).T) + np.dot(1-y, np.log(1-yhat).T))
        return np.squeeze(cost)
  
    def _backprop_this_layer(self, da_curr, z_curr, W_curr, b_curr, A_prev, activation_function):
        if activation_function == 'sigmoid':
            activation_back = self.sigmoid_backward
        elif activation_function == 'relu':
            activation_back = self.relu_backward
        else:
            raise Exception('need sigmoid or relu')
        m = A_prev.shape[1]

        dz_curr = activation_back(da_curr, z_curr)
        dw_curr = np.dot(dz_curr, A_prev.T)/m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True)/m
        da_prev = np.dot(W_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr
    
    def _backward(self, ytrue, ypred, cache):
        grads = {}
        m = ytrue.shape[1]
        da_prev = np.divide(1-ytrue, 1-ypred) - np.divide(ytrue, ypred)
        
        for prev_layer_id, layer in reversed(list(enumerate(self.Architecture))):
            layer_id = prev_layer_id + 1
            activation = layer['activation']

            da_curr = da_prev

            A_prev = cache['A'+str(prev_layer_id)]
            Z_curr = cache['Z'+str(layer_id)]

            W_curr = self.Params['W'+str(layer_id)]
            b_curr = self.Params['b'+str(layer_id)]

            da_prev, dw_curr, db_curr = self._backprop_this_layer(
                da_curr, Z_curr, W_curr, b_curr, A_prev, activation)

            grads["dw"+str(layer_id)] = dw_curr
            grads['db'+str(layer_id)] = db_curr

        return grads
    
    def update(self, grads, learning_rate):
        for layer_id, layer in enumerate(self.Architecture, 1):
            self.Params['W'+str(layer_id)] -= learning_rate * grads['dw'+str(layer_id)]
            self.Params['b'+str(layer_id)] -= learning_rate * grads['db'+str(layer_id)]
        return
    
    def SetThreadId(self,ThreadNetId):
        self.ThreadNetId=ThreadNetId
        return
    
    def SynchThreadLossValues(self,Loss,Epoch):
        global ThreadLock
        global TotalLoss
        global GlobalEpoch
        global Threads
        global CleanUpDone
        
        InsertedGlobal=False        #Will be set to True, when you get lock and updated global struct
        ReadGlobalLoss=False       
        NotReadGlobalCost=True      #Waiting for all threads to insert
        Counter=0
        while (NotReadGlobalCost):
            Counter +=1
            IGotLock=GlobalThread.acquire(blocking=True, timeout=-1)
            if(Epoch % 1 == 0):
                print("Network: {:s} - Epoch: {:05} - AQUIRE={:b} - Counter: {:05}".format(self.Name,Epoch,IGotLock,Counter))
            #Clean Status?
            #print(ThreadStatus)
            #if (IGotLock):
            if (IGotLock & Epoch==GlobalEpoch):
                # Wait until all other threads finsh up Epoch-1
                #
                if not(InsertedGlobal) and (ThreadStatus[self.LockId,4]==0):
                    # Update struct
                    # Initaial status=0 is a condition, -> (CurrentEpoch,MyLoss,Status=1)
                    #
                    # Check for epoch# - should be <= current
                    ThreadStatus[self.LockId,1]=Epoch
                    ThreadStatus[self.LockId,2]=Loss
                    ThreadStatus[self.LockId,3]=1
                    InsertedGlobal=True
                    if(Epoch % 1 == 0):
                        print("Network: {:s} - Epoch: {:05} - Inserted=True - Counter: {:05}".format(self.Name,Epoch,Counter))
                        print("Network: {:s} - Epoch: {:05} - cost: {:.5f}".format(self.Name,Epoch, Loss))
                        print(ThreadStatus)
                elif not(ReadGlobalLoss) and (ThreadStatus[self.LockId,4]==0):
                    # Check all threads inserted
                    # if YES the read out you new GlobalCost
                    # Update bool ReadNewCost = True
                    ThreadAggr=np.sum(ThreadStatus,axis=0)
                    StatusFlags=ThreadAggr[3]
                                        
                    if (StatusFlags/Threads)>=1:
                        #Set the flag that you have
                        #read the global cost
                        ThreadStatus[self.LockId,3]=2
                        Loss=ThreadAggr[2]/Threads
                        ReadGlobalLoss=1
                        # I'm out of here
                        # Set flag that you have exitet synch
                        # Last man standing do the clean-up
                        # Pos 4 holds the flag that you are out of here.
                        ThreadStatus[self.LockId,4]=1
                        if(Epoch % 1 == 0):
                            print("Network: {:s} - Epoch: {:05} - Status->2- Counter: {:05}".format(self.Name,Epoch,Counter))
                            print(ThreadStatus)
                        #GlobalThread.notify_all()
                        #GlobalThread.release()
                        ThreadAggr=np.sum(ThreadStatus,axis=0)
                        StatusAllGone=ThreadAggr[4]
                        #You can leave synch - at least one after you
                        if (StatusAllGone*(Threads-1)==(Threads - 1)):
                            if(Epoch % 1 == 0):
                                print("Network: {:s} - Epoch: {:05} - RELEASE - Counter: {:05}".format(self.Name,Epoch,Counter))
                            GlobalThread.notify_all()
                            GlobalThread.wait()
                            #if GlobalThread.locked():
                            #GlobalThread.release()
                            #sleep(0.05)
                            break
                elif (ThreadStatus[self.LockId,4]==1):
                    ThreadAggr=np.sum(ThreadStatus,axis=0)
                    StatusAllGone=ThreadAggr[4]
                    print("Network: {:s} - Epoch: {:05} - Inside (ThreadStatus[self.LockId,4]==1) - Counter: {:05}".format(self.Name,Epoch,Counter))
                    if (StatusAllGone)==(Threads):
                        #You are the last one to do the clean-up
                        StatusFlags=ThreadAggr[4]
                        ThreadStatus[:,2]=0
                        ThreadStatus[:,3]=0
                        ThreadStatus[:,4]=0
                        if(Epoch % 1 == 0):
                            print("Network: {:s} - Epoch: {:05} - Clean-Up - All Status->0- Counter: {:05}".format(self.Name,Epoch,Counter))
                            print(ThreadStatus)
                        NotReadGlobalCost=False
                        #if GlobalThread.locked():
                        GlobalEpoch +=1
                        GlobalThread.notify_all()
                        GlobalThread.release()
                        break
                if(Epoch % 1 == 0):
                    print("Network: {:s} - Epoch: {:05} - RELEASE - Counter: {:05}".format(self.Name,Epoch,Counter))
                #if GlobalThread.locked():
                GlobalThread.notify_all()
                GlobalThread.wait()
                GlobalThread.release()
            else:
                GlobalThread.notify_all()
                GlobalThread.wait(0.05)
                GlobalThread.wait()
                GlobalThread.release()
                if(Epoch % 1 == 0):
                    print("Network: {:s} - Epoch: {:05} - Waiting...".format(self.Name,Epoch))
                    print(ThreadStatus)
            #GlobalThread.notify_all()       
            #GlobalThread.release()
        print("Network: {:s} - Epoch: {:05} - Exit SynchTreadLossValue...".format(self.Name,Epoch))
        return Loss
    
    def CheckForInfAndNaN(self,loss):
        if (loss == inf):
            loss=100
        if (loss == nan):
            loss=0.01
        return loss

    def fit(self, X, y, epochs, learning_rate, verbose=True, show_loss=True):
        X, y = X.T, y.reshape((y.shape[0],-1)).T
        loss_history, accuracy_history = [], []
        for epoch in tqdm_notebook(range(epochs), total=epochs, unit='epoch'):
            yhat, cache = self._forward(X)
            loss = self.CheckForInfAndNaN(self._criterion(y, yhat))
            loss_history.append(loss)

            yacc = yhat.copy()
            yacc[yacc>0.5] = 1
            yacc[yacc<=0.5] = 0

            accuracy = np.sum(y[0]==yacc[0])/(yacc.shape[1])
            accuracy_history.append(accuracy)

            grads_values = self._backward(y, yhat, cache)

            # Wait for synch between all the threads of self._foward(X) for networks
            # and average 
            # get_synchronized_loss(loss)
            if self.SyncLoss==True:
                loss=self.SynchThreadLossValues(loss,epoch)
            
            self.update(grads_values, learning_rate)
            if(epoch % 1000 == 0):
                if(verbose):
                    print("Network: {:s} - Epoch: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(self.Name,epoch, loss, accuracy))

        
        #fig = plt.figure(figsize=(12,10))
        #plt.plot(range(epochs), loss_history, 'r-')
        #plt.plot(range(epochs), accuracy_history, 'b--')
        #plt.legend(['Training_loss', 'Training_Accuracy'])
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss/Accuracy')
        #plt.show()
        return 

    def predict(self, X):
        print('test')
        yhat, _ = self._forward(X)
        yhat[yhat>0.5]=1
        yhat[yhat<=0.5]=0

        return np.squeeze(yhat)

    def print_time(self,threadName, delay, counter):
        while counter:
            time.sleep(delay)
            print("%s: %s" % (threadName, time.ctime(time.time())))
            counter -= 1
        return


def TestOneTreadOfNetwork(NNTestNumber,LRate,Epochs,NumbNets):
    bc = load_breast_cancer()
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    X_train.shape

    print('-------Start Test case #1 ------- ')
    NN_ARCHITECTURE = [
        {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
        {"input_dim": 32, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
        {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
    ]
    
    #global ThreadLocks
    #global ThreadLoss
    #global Threads

    # Loop on all nets
    ThreadNet0 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-0", 0,2, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet0.SetThreadId(ThreadNet0)
    ThreadNet0.start()
    ThreadNet1 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-1", 1,3, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet1.SetThreadId(ThreadNet1)
    ThreadNet1.start()
    ThreadNet2 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-2", 2,4, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet2.SetThreadId(ThreadNet2)
    ThreadNet2.start()
    ThreadNet4 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-3", 3,5, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet4.SetThreadId(ThreadNet4)
    ThreadNet4.start()
    ThreadNet5 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-4", 4,6, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet5.SetThreadId(ThreadNet5)
    ThreadNet5.start()
    ThreadNet6 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-5", 5,1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet6.SetThreadId(ThreadNet6)
    ThreadNet6.start()
    
    #Threads.append(ThreadNet1)

    #for t in Threads:
    #    t.join()
    #    print ("Exiting Main Thread")
    #print('All treads finished!')

    return

def SquareNumber(ProcessId,Dummy1,Dummy2,send_end):
    for i in range(10):
        
        time.sleep(0.1)
        print("Calculating Squarer for process:, {:1}".format(ProcessId))
        send_end.send('Kjell')
    return

class DeepNeuralNettworkMultiProcesses():
          
    def __init__(self, Architecture,InitBias, InitWeight,ThreadID, Name, LockId,Flag,X_train,y_train,X_test,y_test,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end):
        self.Architecture = Architecture
        self.Bias = InitBias
        self.InitWeight = InitWeight
        self.ThreadID = ThreadID
        self.Name = Name
        self.LockId = LockId
        self.FlagId = Flag
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.Epochs=Epochs
        self.LRate=LRate
        self.verbose=Verbose
        self.ShowLoss=ShowLoss
        self.SyncLoss=SyncLoss
        self.ThreadNetId='ToBeChanged'
        self.send_end=send_end
        self.Params = self._initialize_params(Architecture)
        self.start()
    
        
    def _initialize_params(self, architecture):
        params = {}
        for id_, layer in enumerate(architecture):
            layer_id = id_ + 1

            input_dim = layer['input_dim']
            output_dim = layer['output_dim']

            #Weights multiplied with 0.1 - we try 1/22
            params['W'+str(layer_id)] = np.random.randn(output_dim, input_dim)*self.InitWeight
            params['b'+str(layer_id)] = np.zeros((output_dim, 1))
        return params
    
    def start(self):
        print ("Starting " + self.Name)
        # Get lock to synchronize threads
        #ThreadLock.acquire()
        self.fit(self.X_train, self.y_train, self.Epochs,self.LRate,self.verbose,self.ShowLoss)
        #self.print_time(self.Name, self.Counter, 3)
        # Free lock to release next thread
        #Predict=self.predict(self.X_test)
        Accuracy=self.accuracy(self.X_test,self.y_test)
        #ThreadLock.release()
        self.send_end.send(Accuracy)
        return

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    def relu(self, Z):
        return np.maximum(0, Z)
    #KRC - Cross Entropy
    def crossentropy(self, T, Y):
        E = 0
        for i in range(len(T)):
            if T[i] == 1:
                E -= np.log(Y[i])
            else:
                E -= np.log(1 - Y[i])
        return E

    def sigmoid_backward(self, dA, z_curr):
        sig = self.sigmoid(z_curr)
        return sig*(1-sig)*dA

    def relu_backward(self, dA, z_curr):
        dz = np.array(dA, copy=True)
        dz[z_curr<=0]=0
        return dz
  
    def _forward_prop_this_layer(self, A_prev, W_curr, b_curr, activation_function):
        z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation_function == 'relu':
            activation = self.relu
        elif activation_function == 'sigmoid':
            activation = self.sigmoid
        elif activation_function == 'crossentropy':
            activation = self.crossentropy
        else:
            raise Exception(f"{activation_function} is not supported, Only sigmoid, relu are supported")

        return activation(z_curr), z_curr

    def _forward(self, X):
        cache = {}
        A_current = X
        for layer_id_prev, layer in enumerate(self.Architecture):
            current_layer_id = layer_id_prev+1

            A_previous = A_current
            activation = layer['activation']

            W_curr = self.Params['W'+str(current_layer_id)]
            b_curr = self.Params['b'+str(current_layer_id)]

            A_current, Z_curr = self._forward_prop_this_layer(A_previous, W_curr,
                                                        b_curr, activation)

            cache['A'+str(layer_id_prev)] = A_previous
            cache['Z'+str(current_layer_id)] = Z_curr
            
        return A_current, cache

    def _criterion(self, y, yhat):
        m = yhat.shape[1]
        cost = -1/m * (np.dot(y, np.log(yhat).T) + np.dot(1-y, np.log(1-yhat).T))
        return np.squeeze(cost)
  
    def _backprop_this_layer(self, da_curr, z_curr, W_curr, b_curr, A_prev, activation_function):
        if activation_function == 'sigmoid':
            activation_back = self.sigmoid_backward
        elif activation_function == 'relu':
            activation_back = self.relu_backward
        else:
            raise Exception('need sigmoid or relu')
        m = A_prev.shape[1]

        dz_curr = activation_back(da_curr, z_curr)
        dw_curr = np.dot(dz_curr, A_prev.T)/m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True)/m
        da_prev = np.dot(W_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr
    
    def _backward(self, ytrue, ypred, cache):
        grads = {}
        m = ytrue.shape[1]
        da_prev = np.divide(1-ytrue, 1-ypred) - np.divide(ytrue, ypred)
        
        for prev_layer_id, layer in reversed(list(enumerate(self.Architecture))):
            layer_id = prev_layer_id + 1
            activation = layer['activation']

            da_curr = da_prev

            A_prev = cache['A'+str(prev_layer_id)]
            Z_curr = cache['Z'+str(layer_id)]

            W_curr = self.Params['W'+str(layer_id)]
            b_curr = self.Params['b'+str(layer_id)]

            da_prev, dw_curr, db_curr = self._backprop_this_layer(
                da_curr, Z_curr, W_curr, b_curr, A_prev, activation)

            grads["dw"+str(layer_id)] = dw_curr
            grads['db'+str(layer_id)] = db_curr

        return grads
    
    def update(self, grads, learning_rate):
        for layer_id, layer in enumerate(self.Architecture, 1):
            self.Params['W'+str(layer_id)] -= learning_rate * grads['dw'+str(layer_id)]
            self.Params['b'+str(layer_id)] -= learning_rate * grads['db'+str(layer_id)]
        return
    
    def SetThreadId(self,ThreadNetId):
        self.ThreadNetId=ThreadNetId
        return
    
    def SynchThreadLossValues(self,Loss,Epoch):
        global ThreadLock
        global TotalLoss
        global GlobalEpoch
        global Threads
        global CleanUpDone
        
        InsertedGlobal=False        #Will be set to True, when you get lock and updated global struct
        ReadGlobalLoss=False       
        NotReadGlobalCost=True      #Waiting for all threads to insert
        Counter=0
        while (NotReadGlobalCost):
            Counter +=1
            IGotLock=GlobalThread.acquire(blocking=True, timeout=-1)
            if(Epoch % 1 == 0):
                print("Network: {:s} - Epoch: {:05} - AQUIRE={:b} - Counter: {:05}".format(self.Name,Epoch,IGotLock,Counter))
            #Clean Status?
            #print(ThreadStatus)
            #if (IGotLock):
            if (IGotLock & Epoch==GlobalEpoch):
                # Wait until all other threads finsh up Epoch-1
                #
                if not(InsertedGlobal) and (ThreadStatus[self.LockId,4]==0):
                    # Update struct
                    # Initaial status=0 is a condition, -> (CurrentEpoch,MyLoss,Status=1)
                    #
                    # Check for epoch# - should be <= current
                    ThreadStatus[self.LockId,1]=Epoch
                    ThreadStatus[self.LockId,2]=Loss
                    ThreadStatus[self.LockId,3]=1
                    InsertedGlobal=True
                    if(Epoch % 1 == 0):
                        print("Network: {:s} - Epoch: {:05} - Inserted=True - Counter: {:05}".format(self.Name,Epoch,Counter))
                        print("Network: {:s} - Epoch: {:05} - cost: {:.5f}".format(self.Name,Epoch, Loss))
                        print(ThreadStatus)
                elif not(ReadGlobalLoss) and (ThreadStatus[self.LockId,4]==0):
                    # Check all threads inserted
                    # if YES the read out you new GlobalCost
                    # Update bool ReadNewCost = True
                    ThreadAggr=np.sum(ThreadStatus,axis=0)
                    StatusFlags=ThreadAggr[3]
                                        
                    if (StatusFlags/Threads)>=1:
                        #Set the flag that you have
                        #read the global cost
                        ThreadStatus[self.LockId,3]=2
                        Loss=ThreadAggr[2]/Threads
                        ReadGlobalLoss=1
                        # I'm out of here
                        # Set flag that you have exitet synch
                        # Last man standing do the clean-up
                        # Pos 4 holds the flag that you are out of here.
                        ThreadStatus[self.LockId,4]=1
                        if(Epoch % 1 == 0):
                            print("Network: {:s} - Epoch: {:05} - Status->2- Counter: {:05}".format(self.Name,Epoch,Counter))
                            print(ThreadStatus)
                        #GlobalThread.notify_all()
                        #GlobalThread.release()
                        ThreadAggr=np.sum(ThreadStatus,axis=0)
                        StatusAllGone=ThreadAggr[4]
                        #You can leave synch - at least one after you
                        if (StatusAllGone*(Threads-1)==(Threads - 1)):
                            if(Epoch % 1 == 0):
                                print("Network: {:s} - Epoch: {:05} - RELEASE - Counter: {:05}".format(self.Name,Epoch,Counter))
                            GlobalThread.notify_all()
                            GlobalThread.wait()
                            #if GlobalThread.locked():
                            #GlobalThread.release()
                            #sleep(0.05)
                            break
                elif (ThreadStatus[self.LockId,4]==1):
                    ThreadAggr=np.sum(ThreadStatus,axis=0)
                    StatusAllGone=ThreadAggr[4]
                    print("Network: {:s} - Epoch: {:05} - Inside (ThreadStatus[self.LockId,4]==1) - Counter: {:05}".format(self.Name,Epoch,Counter))
                    if (StatusAllGone)==(Threads):
                        #You are the last one to do the clean-up
                        StatusFlags=ThreadAggr[4]
                        ThreadStatus[:,2]=0
                        ThreadStatus[:,3]=0
                        ThreadStatus[:,4]=0
                        if(Epoch % 1 == 0):
                            print("Network: {:s} - Epoch: {:05} - Clean-Up - All Status->0- Counter: {:05}".format(self.Name,Epoch,Counter))
                            print(ThreadStatus)
                        NotReadGlobalCost=False
                        #if GlobalThread.locked():
                        GlobalEpoch +=1
                        GlobalThread.notify_all()
                        GlobalThread.release()
                        break
                if(Epoch % 1 == 0):
                    print("Network: {:s} - Epoch: {:05} - RELEASE - Counter: {:05}".format(self.Name,Epoch,Counter))
                #if GlobalThread.locked():
                GlobalThread.notify_all()
                GlobalThread.wait()
                GlobalThread.release()
            else:
                GlobalThread.notify_all()
                GlobalThread.wait(0.05)
                GlobalThread.wait()
                GlobalThread.release()
                if(Epoch % 1 == 0):
                    print("Network: {:s} - Epoch: {:05} - Waiting...".format(self.Name,Epoch))
                    print(ThreadStatus)
            #GlobalThread.notify_all()       
            #GlobalThread.release()
        print("Network: {:s} - Epoch: {:05} - Exit SynchTreadLossValue...".format(self.Name,Epoch))
        return Loss
    
    def CheckForInfAndNaN(self,loss):
        if (loss == inf):
            loss=100
        if (loss == nan):
            loss=0.01
        return loss

    def fit(self, X, y, epochs, learning_rate, verbose=True, show_loss=True):
        X, y = X.T, y.reshape((y.shape[0],-1)).T
        loss_history, accuracy_history = [], []
        for epoch in range(epochs):
        #for epoch in tqdm_notebook(range(epochs), total=epochs, unit='epoch'):
            yhat, cache = self._forward(X)
            loss = self.CheckForInfAndNaN(self._criterion(y, yhat))
            loss_history.append(loss)

            yacc = yhat.copy()
            yacc[yacc>0.5] = 1
            yacc[yacc<=0.5] = 0

            accuracy = np.sum(y[0]==yacc[0])/(yacc.shape[1])
            accuracy_history.append(accuracy)

            grads_values = self._backward(y, yhat, cache)

            # Wait for synch between all the threads of self._foward(X) for networks
            # and average 
            # get_synchronized_loss(loss)
            if self.SyncLoss==True:
                loss=self.SynchThreadLossValues(loss,epoch)
            
            self.update(grads_values, learning_rate)
            if(epoch % 1000 == 0):
                if(verbose):
                    print("Network: {:s} - Epoch: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(self.Name,epoch, loss, accuracy))

        
        #fig = plt.figure(figsize=(12,10))
        #plt.plot(range(epochs), loss_history, 'r-')
        #plt.plot(range(epochs), accuracy_history, 'b--')
        #plt.legend(['Training_loss', 'Training_Accuracy'])
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss/Accuracy')
        #plt.show()
        return 

    def accuracy(self,X,y):
        print('Prediction')
        X, y = X.T, y.reshape((y.shape[0],-1)).T
        yhat, _ = self._forward(X)
        yhat[yhat>0.5]=1
        yhat[yhat<=0.5]=0
        accuracy = np.sum(y[0]==yhat[0])/(yhat.shape[1])
        print('Accuracy:{:1}'.format(accuracy)) 
        return accuracy

    def predict(self, X):
        print('test')
        X = X.T
        yhat, _ = self._forward(X)
        yhat[yhat>0.5]=1
        yhat[yhat<=0.5]=0

        return np.squeeze(yhat)

    def print_time(self,threadName, delay, counter):
        while counter:
            time.sleep(delay)
            print("%s: %s" % (threadName, time.ctime(time.time())))
            counter -= 1
        return    

def PlotAccuracy(result_list,type,epochs):
    
    # Loop through the list of thread-results
    # plt.plot(BootstrapRetur[0], BootstrapRetur[1], label='Error train')
    
    label_str='Epochs:' + str(epochs)
    for i in range(len(result_list)):
        if i==0:
            plt.plot(i+1,result_list[i], color='green', marker='o',label=label_str)
        plt.plot(i+1,result_list[i], color='green', marker='o')
    
    pltTitle="Accuracy"
    if type=='Processes':
        plt.xlabel("Process")
        pltFigName=CurrentPath + "/" + 'Processes-' + 'Epochs-' + str(epochs) + '-Accuracy' + '.png'
    else:
        plt.xlabel("Threads")
        pltFigName=CurrentPath + "/" + 'Threads-' + 'Epochs-' + str(epochs) + '-Accuracy' + '.png'
    
    plt.ylim(ymax = 1.05, ymin = 0.85)
    plt.xlim(xmax = 20, xmin = -10)
    
    plt.label='Accuracy'
    plt.ylabel("Accuracy %")
    #Overwrite xlabel if we are plotting processes 
    plt.title(pltTitle)
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()

    return
    
def TestParallelProcessOfNetwork(NNTestNumber,LRate,Epochs,NumbNets):
    from multiprocessing import Process, Pipe, freeze_support
    import os
    import time
    
    RetVal=[]
    bc = load_breast_cancer()
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    X_train.shape

    print('-------Start Test case #1 ------- ')
    NN_ARCHITECTURE = [
        {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
        {"input_dim": 32, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
        {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
    ]
    
    #global ThreadLocks
    #global ThreadLoss
    #global Threads
    
    #freeze_support()
    NumberOfProcesses=os.cpu_count()
    InitBias=0.2   #Initially set to by 0.01?
    InitWeight=1/22   #The weight for balanced dataset 50/50 in {0,1}, should be set with the rule of thumb 1/SQR(n) =1/SQR(512) = 1/22
    ThreadID=1     #Set to loop-id(i) the loop to make them unique
    Name='Process-' #Add lopp-id(i) as postfix to make name unique
    LockId=0       #LockId=loop-id
    Flag=0         #Not used
    LRate=0.0005    #Increment by 0.01
    Verbose=True       
    ShowLoss=True
    SyncLoss=False
     
    Processes=[]
    pipe_list = []
    for i in range(NumberOfProcesses):
        recv_end, send_end = Pipe(False)
        InitBias=InitBias+0.02
        Name=Name+str(i)
        #Epochs +=2000
        #LRate=LRate+0.001
        List=[NN_ARCHITECTURE,InitBias,InitWeight,i,Name,i,Flag, X_train, y_train,X_test, y_test,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end]
        p=Process(target=DeepNeuralNettworkMultiProcesses,args=List)
        #p=Process(target=SquareNumber,args=(i,2,3,send_end))
        #p=Process(target=SquareNumber,args)
        Processes.append(p)
        pipe_list.append(recv_end)
        p.start()
        Name='Process-'
   
    for proc in Processes:
        proc.join()
    result_list = [x.recv() for x in pipe_list]
    print(result_list)
    PlotAccuracy(result_list,'Processes',Epochs)
    print('All processes finished!')        
    return

def TestParallelThreadsOfNetwork(NNTestNumber,LRate,Epochs,NumbNets):
    from multiprocessing import Process, Pipe, freeze_support
    from threading import Thread
    import os
    import time
    
    RetVal=[]
    bc = load_breast_cancer()
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    X_train.shape

    print('-------Start Test case #1 ------- ')
    NN_ARCHITECTURE = [
        {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
        {"input_dim": 32, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
        {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
    ]
    
    #global ThreadLocks
    #global ThreadLoss
    #global Threads
    
    #freeze_support()
    NumberOfThreads=os.cpu_count()
    InitBias=0.2   #Initially set to by 0.01?
    InitWeight=1/22   #The weight for balanced dataset 50/50 in {0,1}, should be set with the rule of thumb 1/SQR(n) =1/SQR(512) = 1/22
    ThreadID=1     #Set to loop-id(i) the loop to make them unique
    Name='Thread-' #Add lopp-id(i) as postfix to make name unique
    LockId=0       #LockId=loop-id
    Flag=0         #Not used
    LRate=0.0005    #Increment by 0.01
    Verbose=True       
    ShowLoss=True
    SyncLoss=False
    
    #Epochs=2000   
    
    Threads=[]
    pipe_list = []
    for i in range(NumberOfThreads):
        recv_end, send_end = Pipe(False)
        InitBias=InitBias+0.02
        Name=Name+str(i)
        #Epochs +=2000
        #LRate=LRate+0.001
        List=[NN_ARCHITECTURE,InitBias,InitWeight,i,Name,i,Flag, X_train, y_train,X_test, y_test,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end]
        p=Thread(target=DeepNeuralNettworkMultiProcesses,args=List)
        #p=Process(target=SquareNumber,args=(i,2,3,send_end))
        #p=Process(target=SquareNumber,args)
        Threads.append(p)
        pipe_list.append(recv_end)
        p.start()
        Name='Thread-'
   
    for proc in Threads:
        proc.join()
    result_list = [x.recv() for x in pipe_list]
    print(result_list)
    PlotAccuracy(result_list,'Threaded',Epochs)
    print('All Threads finished!')
            
    return

def TestMultiThreadedNerworks(NNTestNumber,LRate,Epochs,NumbNets):
    bc = load_breast_cancer()
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    X_train.shape

    print('-------Start Test case #1 ------- ')
    NN_ARCHITECTURE = [
        {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
        {"input_dim": 32, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
        {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
    ]
    
    # Loop on all nets
    ThreadNet1 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-1", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet2 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-2", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet3 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-3", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet4 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-4", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet5 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-5", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet6 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-6", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet7 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-7", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet8 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-8", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet9 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-9", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet10 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-10", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet11 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-11", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet12 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-12", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet13 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-13", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet14 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-14", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet15 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-15", 1, X_train, y_train,Epochs,LRate,True,True)
    ThreadNet16 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-16", 1, X_train, y_train,Epochs,LRate,True,True)
    
    ThreadNet1.start()
    ThreadNet2.start()
    ThreadNet3.start()
    ThreadNet4.start()
    ThreadNet5.start()
    ThreadNet6.start()
    ThreadNet7.start()
    ThreadNet8.start()
    ThreadNet9.start()
    ThreadNet10.start()
    ThreadNet11.start()
    ThreadNet12.start()
    ThreadNet13.start()
    ThreadNet14.start()
    ThreadNet15.start()
    ThreadNet16.start()

    Threads.append(ThreadNet1)
    Threads.append(ThreadNet2)
    Threads.append(ThreadNet3)
    Threads.append(ThreadNet4)
    Threads.append(ThreadNet5)
    Threads.append(ThreadNet6)
    Threads.append(ThreadNet7)
    Threads.append(ThreadNet8)
    Threads.append(ThreadNet9)
    Threads.append(ThreadNet10)
    Threads.append(ThreadNet11)
    Threads.append(ThreadNet12)
    Threads.append(ThreadNet13)
    Threads.append(ThreadNet14)
    Threads.append(ThreadNet15)
    Threads.append(ThreadNet16)
    
    # predictions = net1.predict(X_test.T)

    # Create new threads
    #thread1 = myThread(1, "Thread-1", 1)
    #thread2 = myThread(2, "Thread-2", 2)
    # Start new Threads
    #thread1.start()
    #thread2.start()
    # Add threads to thread list
    #threads.append(thread1)
    #threads.append(thread2)
    # Wait for all threads to complete
    
    for t in Threads:
        t.join()
        print ("Exiting Main Thread")
    print('All treads finished!')


    return

def TestCases3DNeuralNetwork(NNTestNumber,LRate,Epochs):
    bc = load_breast_cancer()
    bc = load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    X_train.shape

    if NNTestNumber == 1:
        print('-------Start Test case #1 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
            {"input_dim": 32, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
            {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
        ]
        net1 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1, "Thread-1", 1)
        net1.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net1.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #1 ------- ')
    elif (NNTestNumber==2):
        print('-------Start Test case #2 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30,  "output_dim": 32,  "activation": "relu"}, # Input Layer
            {"input_dim": 32,  "output_dim": 32,  "activation": "relu"},# Hidden Layer -- 1
            {"input_dim": 32,  "output_dim": 32,  "activation": "relu"},# Second Hidden Layer
            {"input_dim": 32,  "output_dim": 32,  "activation": "relu"},# Third Hidden Layer
            {"input_dim": 32,  "output_dim": 1,   "activation": "sigmoid"},# Output Layer
        ]
        net2 = DeepNeuralNettwork(NN_ARCHITECTURE)
        net2.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net2.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #2 ------- ')
    elif (NNTestNumber==3):
        print('-------Start Test case #3 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30,  "output_dim": 32,  "activation": "relu"}, # Input Layer
            {"input_dim": 32,  "output_dim": 16,  "activation": "relu"},# Hidden Layer -- 1
            {"input_dim": 16,  "output_dim": 16,  "activation": "relu"},# Second Hidden Layer
            {"input_dim": 16,  "output_dim": 16,  "activation": "relu"},# Third Hidden Layer
            {"input_dim": 16,  "output_dim": 1,   "activation": "sigmoid"},# Output Layer
        ]
        net3 = DeepNeuralNettwork(NN_ARCHITECTURE)
        net3.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net3.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #3 ------- ')
    elif (NNTestNumber==4):
        print('-------Start Test case #4 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30,  "output_dim": 32,  "activation": "relu"}, # Input Layer
            {"input_dim": 32,  "output_dim": 16,  "activation": "relu"},# Hidden Layer -- 1
            {"input_dim": 16,  "output_dim": 1,   "activation": "sigmoid"},# Output Layer
        ]
        net4 = DeepNeuralNettwork(NN_ARCHITECTURE)
        net4.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net4.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #4 ------- ')
    elif (NNTestNumber==5):
        print('-------Start Test case #5 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
            {"input_dim": 32, "output_dim": 64, "activation": "relu"},# Hidden Layer -- 1
            {"input_dim": 64, "output_dim": 64, "activation": "relu"},# Second Hidden Layer
            {"input_dim": 64, "output_dim": 32, "activation": "relu"},# Third Hidden Layer
            {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
        ]
        net4 = DeepNeuralNettwork(NN_ARCHITECTURE)
        net4.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net4.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #5 ------- ')
    elif (NNTestNumber==6):
        print('-------Start Test case #6 ------- ')
        NN_ARCHITECTURE = [
            {"input_dim": 30, "output_dim": 32, "activation": "relu"}, # Input Layer
            {"input_dim": 32, "output_dim": 84, "activation": "relu"},# Hidden Layer -- 1
            {"input_dim": 84, "output_dim": 84, "activation": "relu"},# Hidden Layer -- 2
            {"input_dim": 84, "output_dim": 84, "activation": "relu"},# Hidden Layer -- 3
            {"input_dim": 84, "output_dim": 32, "activation": "relu"},# Hidden Layer -- 4
            {"input_dim": 32, "output_dim": 1,  "activation": "sigmoid"},# Output Layer
        ]
        net4 = DeepNeuralNettwork(NN_ARCHITECTURE)
        net4.fit(X_train, y_train, epochs=Epochs, learning_rate=LRate, verbose=True, show_loss=True)
        predictions = net4.predict(X_test.T)
        #accuracy_score(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print('-------Finished Tescase #6 ------- ')
    return

def TestDeepNeuralNetwork():
    
    #######   TEST 3-D FFNN  ###############
    
    TestParallelThreadsOfNetwork(1,0.0003,1001,1)
    # TestParallelThreadsOfNetwork(1,0.0003,2001,1)
    # TestParallelThreadsOfNetwork(1,0.0003,4001,1)
    # TestParallelThreadsOfNetwork(1,0.0003,8001,1)
    # TestParallelThreadsOfNetwork(1,0.0003,16001,1)
    
    TestParallelProcessOfNetwork(1,0.0003,1001,1)
    # TestParallelProcessOfNetwork(1,0.0003,2001,1)
    # TestParallelProcessOfNetwork(1,0.0003,4001,1)
    # TestParallelProcessOfNetwork(1,0.0003,8001,1)
    # TestParallelProcessOfNetwork(1,0.0003,16001,1)
    
    return
    
  


def Main():
    #This module is central for running an testing both Project 1 & 2
    #
    # Samples = 2000
    # x = np.random.uniform(0,1,size=Samples)
    # y = np.random.uniform(0,1,size=Samples)

    # #Scale the data based on subracting the mean
    # x=SubMean(x)
    # y=SubMean(y)

    # #Part-A
    # # Test for plynomials up to 5'th degree by using the Franke's function
    # # Regression analysis using OLS/MSE, to find confidence intervals,
    # # varances, MSE. Use scaling of the data (subtractiong mean) and add noise.
    # #
    # # Test ut saving a files with .png extension
    global CurrentPath
    CurrentPath=MakeDirAndReturnPath("/Plots")

    # PolyDim=20
    # z = FrankeFunction(x, y)
    # z = AddNoise(z,Samples)
    # X = DesignMatrix(x,y,PolyDim)
    # Lamdas = [0.0001, 0.001, 0.01, 0.1, 1, 2]
    # Bootstraps=1000
    # Folds=5
    
    # Set to 15 to getter a smoother curve
    #OLSConfidenceInterval (x,y,Samples,X,PolyDim)
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Ridge")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Lasso")
   
    #Part-B
    # Bias-Variance crossvalidation using OLS with Bootstrapping
    #
    #KfoldCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")

    #Part-C
    #Compare K-fold and Bootstrap on the OLS/MSE
    #KfoldCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"OLS")

    #Part-D
    #Use Matrix Inversion for 
    # Make an analysis on lamdas, by using bootstrap
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"Ridge")

    #Part-E
    #Use Matrix Inversion for 
    # Make an analysis on lamdas, by using bootstrap
    # Compare and analyze the three methods (OLS,Ridge,Lasso)  
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"Lasso")

    #Part-F
    # Set up a routine to read REAL data.
    #IntroRealData()

    #Part-G
    # Set up a routine to read REAL data.
    #RealDataAnalyzis(PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples)
    
    #Project 2

    #TestLinearRegression()
    #TestDeepNeuralNetwork()
    #MNISTLogisticRegression()
    
    #LogisticRegression()
    #LogisticRegNormEqu()
    #TestSGDLinearReg()
    #FFNNRegression()
    #NNMnist()
    #NN()

    # Project #3
    TestDeepNeuralNetwork() 

    return

    # Call Main Module and mark selektet Parts of the 
    # program to run

if __name__ == '__main__':
    freeze_support()
    Main()    