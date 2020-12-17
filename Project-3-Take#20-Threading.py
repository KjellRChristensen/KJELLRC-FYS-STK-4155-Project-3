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

# Definition of the Franke's function as in the text.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Define DesingMatrix, to run with Numba and in parallel, if posible
#@njit(parallel=True)
def DesignMatrix(x,y,n):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)
        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements/columns in beta                                                               
        X = np.ones((N,l))

        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)
        return X

#Add noise to Z
def AddNoise(z,n):
    return z + (0.005*np.random.randn(n))

#Find beta, by using matrix invers functionality
#For Franke's we are passing in Z for the "y" vector.
def BetaMatrixInv(X,y):
    Beta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return (X @ Beta)

def ConfidenceIntervals(VarPredict,MeanPredict,n):
    # (80% =Z=1.282), (85%=Z=1.440),(90%=Z=1.645),
    # (95% =Z=1.960), (99%=Z=2.576),(99.5%=Z=2.807 (99.9%=Z=3.291)
    # We use confidence interval 99.5%
    z=2.807
    print("99.5% Confidence interval")
    print("Mean +- Z*(StandardDev /SQR(n)")
    print("Say for 99.5% we have")        
    print("Mean +- 2.807*(SQR(VarPredict)/SQR(n))")
    print(MeanPredict-(z*np.sqrt(VarPredict)/np.sqrt(n)))
    print(MeanPredict+(z*np.sqrt(VarPredict)/np.sqrt(n)))
    #Define Mean Square Error (OLS)
    return

def MSE(YData,YPredict):
    DimY=len(YData)
    MSE= (YData -YPredict).T@(YData - YPredict)/DimY
    return MSE

def GetVariance(YPredict):
    return np.sum((YPredict - np.mean(YPredict))**2)/np.size(YPredict)

def GetMean(YPredict):
    return  np.mean(YPredict, axis=0)

def SubMean(YPredict):
    YPredict = YPredict - np.mean(YPredict, axis=0)
    return YPredict

def GetR2 (YData,YPredict):
    YDataMean=np.mean(YData)    
    return 1-((YData-YPredict).T@(YData-YPredict))/((YData-YDataMean).T@(YData-YDataMean))

def SplitDataSet(x_,y_,z_,i):
    #Quick way to delete and extract 
    #elements from list is by using np
    #delete & take
    #Try np.setdiff1d(a,b)
    x_learn=np.delete(x_,i)
    y_learn=np.delete(y_,i)
    z_learn=np.delete(z_,i)
    x_test=np.take(x_,i)
    y_test=np.take(y_,i)
    z_test=np.take(z_,i)
    return (x_learn,y_learn,z_learn,x_test,y_test,z_test)

def BootStrap(X,Degrees,Lamdas,Bootstraps,x,y,z,Samples,ReggrType):
    # Bootstrap also used for OLS, Ridge & Lasse and need a "switch (if statement)"
    # -Select # of bootstraps with replacement.
    # We need to switch on Bootstrap-values of Lamdas
    # x_train, x_test, y_train, y_test, z_train, z_test = SplitDataSet(x, y, z, test_size=0.2, shuffle=True)
    # Need modificatino to shuffle
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
    
    #Dette må parameteriseres
    
    s = (x_test.shape[0],Bootstraps)
    z_test1 = np.zeros(s)
    s = (x_train.shape[0],Bootstraps)
    z_train1 = np.zeros(s)

    for i in range(Bootstraps):
        z_test1[:,i]=z_test

    if (ReggrType=="OLS"):
        DegreeOrLamda=Degrees
    else:
        #Need to extract # of elements in the set
        DegreeOrLamda=len(Lamdas)

    #Setting up the stuct based on # of Degrees and
    #length of lampda elements passed in.
    error_test = np.zeros(DegreeOrLamda)
    bias___ = np.zeros(DegreeOrLamda)
    variance___ = np.zeros(DegreeOrLamda)
    polylamda = np.zeros(DegreeOrLamda)
    error_train = np.zeros(DegreeOrLamda)     
    
    for CurrDegreeOrLamda in range(DegreeOrLamda):
        z_pred = np.empty((z_test.shape[0],Bootstraps))
        z_pred_train = np.empty((z_train.shape[0],Bootstraps))
        
        # If Regression is not OLS, then we need
        # to preserve en loop through the values of the
        # lamdas - not the index
        # 
        if (ReggrType!="OLS"):
            CurrLamda=Lamdas[CurrDegreeOrLamda]
      
        for i in range(Bootstraps):
            xResample, yResample, zResample = resample(x_train, y_train, z_train)
            z_test1[:,i]=z_test
            z_train1[:,i] = zResample
           
            XTrain = DesignMatrix(xResample,yResample,CurrDegreeOrLamda)
            XTest= DesignMatrix(x_test,y_test,CurrDegreeOrLamda)

            if (ReggrType == "OLS"):
                print("Bootstrap based on OLS")
                #Beta=linear_model.LinearRegression(fit_intercept=False).fit(X,z)
                z_pred[:, i] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTrain).ravel()
            elif (ReggrType == "Ridge"):
                print("Bootstrap based on Ridge")
                #Beta=linear_model.Ridge(alpha=Lamdas)
                z_pred[:, i] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTrain).ravel()
            else:
                print("Bootstrap based on Lasso")
                #Beta=linear_model.Lasso(alpha=Lamdas)
                z_pred[:, i] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTrain).ravel()
        
        if (ReggrType=="OLS"):
            polylamda[CurrDegreeOrLamda]    = CurrDegreeOrLamda
            error_test[CurrDegreeOrLamda]   = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[CurrDegreeOrLamda]      = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[CurrDegreeOrLamda]  = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[CurrDegreeOrLamda]  = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        else:
            polylamda[Lamdas.index(CurrLamda)]   = CurrLamda
            error_test[Lamdas.index(CurrLamda)]  = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[Lamdas.index(CurrLamda)]     = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[Lamdas.index(CurrLamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[Lamdas.index(CurrLamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        
        print(CurrDegreeOrLamda)
        print(error_test)
        print(bias___)
        print(variance___)
        print(bias___+variance___)
    return (polylamda,error_train,error_test, bias___,variance___ )

def KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # K-Fold used for OLS,Ridge & Lasse and need a "switch (if statement)"
    # to select the specific model configuration.
    # k,x,y,z,m,model
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
    
    j=np.arange(Samples)
    np.random.shuffle(j)
    n_k=int(Samples/Folds)
    

    #Dette må parameteriseres
    
    s = (x_test.shape[0],Folds)
    z_test1 = np.zeros(s)
    s = (x_train.shape[0],Folds)
    z_train1 = np.zeros(s)

    for i in range(Folds):
        z_test1[:,i]=z_test

    if (ReggrType=="OLS"):
        DegreeOrLamda=Degrees
    else:
        #Need to extract # of elements in the set
        DegreeOrLamda=len(Lamdas)

    #Setting up the stuct based on # of Degrees and
    #length of lampda elements passed in.
    error_test = np.zeros(DegreeOrLamda)
    bias___ = np.zeros(DegreeOrLamda)
    variance___ = np.zeros(DegreeOrLamda)
    polylamda = np.zeros(DegreeOrLamda)
    error_train = np.zeros(DegreeOrLamda)     
    
    for CurrDegreeOrLamda in range(DegreeOrLamda):
        z_pred = np.empty((z_test.shape[0],Folds))
        z_pred_train = np.empty((z_train.shape[0],Folds))
        
        # If Regression is not OLS, then we need
        # to preserve en loop through the values of the
        # lamdas - not the index
        # 
        if (ReggrType!="OLS"):
            CurrLamda=Lamdas[CurrDegreeOrLamda]

        # Change Bottstraps with Folds
        for Fold in range(Folds):
            i=Fold
            # xResample,yResample,zResample,xTest,yTest,zTest=train_test_split(x, y, z, test_size=0.2, shuffle=True)
            xResample,yResample,zResample,xTest,yTest,zTest=SplitDataSet(x, y, z,j[i*n_k:(i+1)*n_k])
            
            z_test1[:,Fold]=zTest
            z_train1[:,Fold] = zResample
            
            XTrain = DesignMatrix(xResample,yResample,CurrDegreeOrLamda)
            XTest= DesignMatrix(x_test,y_test,CurrDegreeOrLamda)

            if (ReggrType == "OLS"):
                print("K-Fold based on OLS")
                Betas=linear_model.LinearRegression(fit_intercept=False).fit(X,z)
                z_pred[:, Fold] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTrain).ravel()
            elif (ReggrType == "Ridge"):
                print("K-Fold based on Ridge")
                Betas=linear_model.Ridge(alpha=Lamdas)
                z_pred[:, Fold] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTrain).ravel()
            else:
                print("K-Fold based on Lasso")
                Betas=linear_model.Lasso(alpha=Lamdas)
                z_pred[:, Fold] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTrain).ravel()
                
        if (ReggrType=="OLS"):
            polylamda[CurrDegreeOrLamda]    = CurrDegreeOrLamda
            error_test[CurrDegreeOrLamda]   = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[CurrDegreeOrLamda]      = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[CurrDegreeOrLamda]  = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[CurrDegreeOrLamda]  = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        else:
            polylamda[Lamdas.index(CurrLamda)]   = CurrLamda
            error_test[Lamdas.index(CurrLamda)]  = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[Lamdas.index(CurrLamda)]     = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[Lamdas.index(CurrLamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[Lamdas.index(CurrLamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        
        print(CurrDegreeOrLamda)
        print(error_test)
        print(bias___)
        print(variance___)
        print(bias___+variance___)
    
        #error_test = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
        #bias___ = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
        #variance___ = np.mean( (z_pred - np.mean(z_pred, axis=1, keepdims=True))**2 )
        #error_train = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        #(error_test, bias___,variance___ , error_train, R2_t, np.std(Betas, axis = 0), np.mean(Betas, axis = 0))



    return (polylamda,error_train,error_test, bias___,variance___ )

def OLSConfidenceInterval (x,y,n,X,PolyDimRange):
    # This is the main program for Part B
    # Loop through all relevant plynomials (2-5) and for each loop:
    # -Create the Design Matrix for current polinomial
    # -
    # -Loop through a selected points of 
    for dim in range(PolyDimRange-1):
        X1=DesignMatrix(x,y,dim+2)
        X11= pd.DataFrame(X1)
        print(X11)
        Z1=FrankeFunction(x,y)
        Z1 = AddNoise(Z1,n)
        Z1Tilde=BetaMatrixInv(X1,Z1)
        #KRC - Dette skal puttes inn i en strukt for analyse - print out!
        # The mean squared error 
        #ßprint("Polynomial DIM    : %")
        print("Ploynomial Dim    : %d" % int(dim+2))
        print("Mean value        : %.2f" % np.mean(Z1Tilde))
        print("Mean(Code) value  : %.2f" % GetMean(Z1Tilde))
        print("Scaling YPredict - mean")
        #print(SubMean(Z1Tilde))
        #print("Variance          : %.2f" % np.var(Z1,Z1Tilde))
        print("Mean squared error: %.5f" % mean_squared_error(Z1, Z1Tilde))
        print("MSE(code) value")
        print(MSE(Z1,Z1Tilde))   
        print("Variance(code):")
        print(GetVariance(Z1Tilde))                       
        # Explained variance score: 1 is perfect prediction                                 
        print('R2 Variance score: %.5f' % r2_score(Z1, Z1Tilde))
        print("R2(code) value")
        print(GetR2(Z1,Z1Tilde))
        VarYTilde=GetVariance(Z1Tilde)
        ConfidenceIntervals(VarYTilde,np.mean(Z1Tilde),len(Z1Tilde.shape))
        # Loop and add to dictionary
    return

def KfoldCrossValidation(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part b)
    KFoldRetur=KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    plt.plot(KFoldRetur[0], KFoldRetur[2], label='Error test')
    plt.plot(KFoldRetur[0], KFoldRetur[3], label='Bias')
    plt.plot(KFoldRetur[0], KFoldRetur[4], label='Variance')
    #plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    #plt.plot(f[0], f[2], label='bias')
    #plt.plot(f[0], f[3], label='Variance')
    if (ReggrType=="OLS"):
        plt.xlabel("Polynomial")
        pltTitle=str(ReggrType) + ":K-Fold" + ":Samples:" + str(Samples) + ":Folds:" + str(Folds)
        pltFigName=str(ReggrType) +":K-Fold" + ":Samples:" + str(Samples) + ":Folds:" + str(Folds) + ".png"
    else:
        plt.xlabel("Lamdas") 
        pltTitle=str(ReggrType)+ ":K-Fold" + ":Samples:" + str(Samples) + "Lamdas:"
        pltFigName=str(ReggrType)+ ":K-Fold" + ":Samples:" + str(Samples) + "Lamdas:" + ".png"
    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()
    return

def BootStrapCrossValidation(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part b)
    BootstrapRetur=BootStrap(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(BootstrapRetur[0], BootstrapRetur[1], label='Error train')
    plt.plot(BootstrapRetur[0], BootstrapRetur[2], label='Error test')
    plt.plot(BootstrapRetur[0], BootstrapRetur[3], label='Bias')
    plt.plot(BootstrapRetur[0], BootstrapRetur[4], label='Variance')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[1], label='Error train')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[2], label='bias')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[3], label='Variance')
    if (ReggrType=="OLS"):
        plt.xlabel("Polynomial")
        pltTitle=str(ReggrType) + ":Bootstrap" + ":Samples:" + str(Samples) + ":Polynomial:" + str(Degrees)
        pltFigName=CurrentPath + "/" + str(ReggrType) +"-Bootstrap" + "-Samples" + str(Samples) + "-Polynomial-" + str(Degrees) + ".png"
    else:
        plt.xlabel("Lamdas") 
        pltTitle=str(ReggrType)+ ":Bootstrap" + ":Samples:" + str(Samples) + "Lamdas:"
        pltFigName=CurrentPath + "/" + str(ReggrType)+ "-Bootstrap" + "-Samples-" + str(Samples) + "Lamdas-"+".png"

    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()
    return 

def RigdeRegression(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part d)
      # This is the "main" program for Part b)
    KFoldRetur=KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    plt.plot(KFoldRetur[0], KFoldRetur[2], label='Error test')
    plt.plot(KFoldRetur[0], KFoldRetur[3], label='Bias')
    plt.plot(KFoldRetur[0], KFoldRetur[4], label='Variance')
    #plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    #plt.plot(f[0], f[2], label='bias')
    #plt.plot(f[0], f[3], label='Variance')
    pltTitle=str(ReggrType) + ":Samples:" + str(Samples) + ":Lamda:" + str(Lamdas)
    plt.xlabel("Polynomial")
    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.show()
    return 

def LassoRegression(x,y,z,k,SetOfLamdas):
    # This is the "main" program for Part e)
    # NB
    # Give a critical discussion of the three methods and a 
    # judgement of which model fits the data best.
    #
    SetOfLamdas=[0.001, 0.01, 0.1, 1, 2]
    error = np.zeros(len(SetOfLamdas))
    bias = np.zeros(len(SetOfLamdas))
    variance = np.zeros(len(SetOfLamdas))
    polylamda = np.zeros(len(SetOfLamdas))
    for lamda in SetOfLamdas: 
        lamda_fold =KFold(Folds,x,y,z,Dim,Lamda,"Lasso")
        error_ = lamda_fold[0]
        bias_ = lamda_fold[2]
        #print(bias_)
        variance_ = lamda_fold[3]
       # print('AAA')
        #print(SetOfLamdas.index(lamda))
        polylamda[SetOfLamdas.index(lamda)] = lamda
        error[SetOfLamdas.index(lamda)] = error_
        bias[SetOfLamdas.index(lamda)] = bias_
        variance[SetOfLamdas.index(lamda)] = variance_
        #plt.plot(f[0], f[1], label='Error')
        #plt.plot(f[0], f[2], label='bias')
        #plt.plot(f[0], f[3], label='Variance')
        #plt.legend()
        #plt.show()
    return

def IntroRealData():
    #############################################################
    ######                      PART F                     ######
    #############################################################
    # Import and prepare data analysis
    # 
    #
    # Load the terrain                                                                                  
    # terrain = imread('SRTM_data_Norway_1.tif')
    terrain = imread('SRTM_data_Norway_2.tif')

    # just fixing a set of points
    N = 1000
    m = 5 # polynomial order                                                                            
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels                                                                      
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)
    # Note the use of meshgrid
    # NB-KRCC
    # X = create_X(x_mesh, y_mesh,m)
    # you have to provide this function

    # Show the terrain                                                                                  
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

def RealDataAnalyzis(PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples):
    #############################################################
    ######                      PART G                    ######
    #############################################################
    # Import and prepare data analysis
    # Load the terrain                                                                                  
    # terrain = imread('SRTM_data_Norway_1.tif')
    terrain = imread('SRTM_data_Norway_2.tif')

    # just fixing a set of points
    N = 1000                                                                  
    terrain = terrain[:N,:N]

    #Extract x and y from topographic data (image)
    x=terrain[0]
    y=terrain[1]
    #Scale x & y's
    x=SubMean(x)
    y=SubMean(y)

    #PolyDim=10
    Samples=N
   
    z = FrankeFunction(x, y)

    # Plot the surface
    # z need to be in two dim, to be able to plot?

    #fig = plt.figure()
    #ax = fig.gca(projection='3d') 

    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Plot the surface.surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()

    # z = AddNoise(z,Samples)
    X = DesignMatrix(x,y,PolyDim)
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Ridge")
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Lasso")
    
    # Creates mesh of image pixels                                                                      
    x_m = np.linspace(0,1, np.shape(terrain)[0])
    y_m = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x_m,y_m)
    

    # Show the terrain       
    fig = plt.figure()
    ax = fig.gca(projection='3d')                                                                           
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

def OLSRigdeLassoRealData():
    # Run selected samples of data points
    # and run analyzis of all three methods 
    # and present this i documentation and 
    # plots (located in plots folder on GitHub repository)
    
    return

############ PROJECT 2 ###########
##################################

def GetBatchSize (N,ProcOfN):
    # Based on the HW you are running test on, you should set
    # the size and "fraction" of your test and training sets 
    # according to the size on N (totalt # of data sets)
    # It is important to have a random selections of buckets ot
    # of the total samples.
    #
    MyHW = 10000
    if (N >=MyHW):
        # We select only ProcOfN from the total samples and plit
        #
        BucketSize= divmod(MyHW * ProcOfN)   
    else:
        BucketSize= divmod(N*ProcOfN)  
    return BucketSize

def GetRandomIndex(Samples):
    return np.random.randint(Samples) 


def LinearRegNormEqu(X,y,XBetha):
    # For testing out the different versions of linear regressen
    # we construct a "random" Y = ax + b 
    #
    # Normal Equation
    # Finds the " exact" mathematical solution
    theta_best = np.linalg.inv(XBetha.T.dot(XBetha)).dot(XBetha.T).dot(y)

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    y_predict

    #Plotting the points in blue
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])

    #Plotting the Normal Equation line in red
    plt.plot(X_new, y_predict, "r-")
    
    save_fig("data_plot_normal_equation")
    plt.show()
    return

def plot_gradient_descent(X,y,theta, eta, theta_path=None):
        
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    global theta_path_mgd

    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 2000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        #if theta_path is not None:
        #    theta_path.append(theta)
        theta_path_mgd.append(theta)
    return

def LinearRegFullGD(XBeta,y,Theta):
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    np.random.seed(42)
    theta = np.random.randn(2,1)  # random initialization

    plt.figure(figsize=(10,4))
    plt.figure()
    #Eta from topp 0.02, 0.1, 0.5,
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    eta=0.01
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

    plt.subplot(231); plot_gradient_descent(X,y,theta, eta=0.01)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    eta=0.03
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    plt.subplot(232); plot_gradient_descent(X,y,theta, eta=0.03, theta_path=theta_path_bgd)
    eta=0.06
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    plt.subplot(233); plot_gradient_descent(X,y,theta, eta=0.06)
    eta=0.08
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    plt.subplot(234); plot_gradient_descent(X,y,theta, eta=0.08)
    #plt.ylabel("$y$", rotation=0, fontsize=18)
    eta=0.1
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    plt.subplot(235); plot_gradient_descent(X,y,theta, eta=0.1,theta_path=theta_path_bgd)
    eta=0.14
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    plt.subplot(236); plot_gradient_descent(X,y,theta, eta=0.14)
    plt.title("Linear regresseion Numbers:100 Iteration:1000")
    # plt.set_title("Linear regresseion Numbers:100 Iteration:1000")
    save_fig("gradient_descent_plot")
    plt.show()
    return

def LinearRegMiniBatch(X_b,y,m):

    theta_path_mgd = []

    n_iterations = 50
    minibatch_size = 20

    np.random.seed(42)
    theta = np.random.randn(2,1)  # random initialization

    t0, t1 = 200, 1000
    def learning_schedule(t):
        return t0 / (t + t1)

    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
        
    return theta_path_mgd

def LinearRegStochasticGD(Samples):
    # Part 2.A
    # Write a Stochastic Gradient Descent algoritm to replace the cost function
    # for OLS and Ridge methods. The previous code for matrix invesion to find betha,
    # will be replaced med SDGM.
    # Routine will use Franke's data passed in as "z"
    # 1) Ramdomized weight mattrix as a starting point.
    # 2) Calculate the differential, with respect to the cost function.
    # 2.1 We need a cost funcion for both OLS/MSE & Ridge
    # 3) Kalkulate for one instance at a time and correct weight.
    # 4) Repeate until you have reaced a stopin point, or error ≈ 0.
    # 5) Tune hyperparameters and compare to MSE and Ridge.
    # Sjekk ut random_indices = np.random.choice(indices, size=5) for bruk
    # 6) Set up mini batches, based on the total datapoints.
    # 7) Gjør analyse av size på Minibatch mot total tid mot presisjon
    # 8) Benytt adagrad (finn eller lage en python fun())

    # som stokastisk selector.

    Samples=100
    #Epoch = 50
    Epochs=100
    p=1
    
    np.random.seed(1776)

    # Learning rate parameters
    # Test different values for t0 & t1
    t0, t1 = 5, 50   
    def LearningRate(t):
        return t0 / (t + t1)

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    #Check with the best Theta

    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_path_sgd = []
    m = len(X_b)

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)    
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    yPredict = X_new_b.dot(theta_best)
    print(yPredict)

    StepSize = 0.1 #Will be adjusted proportional to the gradient.
    Weights = np.random.randn(2,1)

    #Loop over # of Epochs
    for CurrEpoch in range(Epochs):
        #Loop through 1 full Epoch (Samples)
        for i in range(Samples):                
            #MiniBatch=GetRandomBatches(Samples)
            if CurrEpoch == 0 and i < 10:                   
                YPredict = X_new_b.dot(Weights)             
                style = "b-" if i > 0 else "r--"           
                plt.plot(X_new, YPredict, style)     
            SelectedIndex=GetRandomIndex(Samples)
            xSelected = X_b[SelectedIndex:SelectedIndex+1]
            ySelected = y[SelectedIndex:SelectedIndex+1]
            Gradient = (2 * xSelected.T.dot(xSelected.dot(Weights) - ySelected))
            StepSize = LearningRate(CurrEpoch * m + i)
            Weights = Weights - StepSize * Gradient
            theta_path_sgd.append(Weights)   

            #gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            #eta = learning_schedule(epoch * m + i)
            #theta = theta - eta * gradients
            #theta_path_sgd.append(theta)

    #print(Weights)
    #plt.plot(X, y, "b.")                                 
    #plt.xlabel("$x_1$", fontsize=18)                
    #plt.ylabel("$y$", rotation=0, fontsize=18)   
    #plt.axis([0, 2, 0, 15])                      
    #save_fig("sgd_plot")                        
    #plt.show()      
    return theta_path_sgd

def TestLinearRegression():
    # Generate random points
    X= 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    XBetha = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    
    #Normal Equation
    #-OK-KRC LinearRegNormEqu(X,y,XBetha)



    Eta=0.1
    n_iterations = 1000
    Instances = 100

    Theta = np.random.randn(2,1)  # random initialization

    for iteration in range(n_iterations):
        Grad = 2/Instances * XBetha.T.dot(XBetha.dot(Theta) - y)
        Theta = Theta - Eta * Grad
    
    # Test batch/full gradient descent
    # for different learning rates
    theta_path_mgd = []

    #KRCC Denne rutinen skal splittes op
    theta_path_bgd=LinearRegFullGD(XBetha,y,Theta)

    theta_path_sgd=LinearRegStochasticGD(Instances)

    #Map the descent to min for the tree methods
    theta_path_mgd=LinearRegMiniBatch(XBetha,y,Instances)


    #theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)

    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    save_fig("gradient_descent_paths_plot")
    plt.show()
    return

def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

def ConfusionMatrix(YTest,Predictions):
    cm = metrics.confusion_matrix(YTest, Predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);
    return

def GetStochasticIndecesOfN (N,BucketSize):
    #Check if buckeds can have the same test data
    #selveral times?
    #
    #SuffeldIndices[] =np.random(1,N) 
    return

def GetRandomBatches(Samples):
    # Split the samples into buckets and
    # process all the buckets with the current
    # epock.
    #
    NoOfSamples = len(Samples)
    BatchSize=GetBatchSize()
    random.shuffle(Samples)
    MiniBatches = [Samples[k:k+BatchSize]
                for k in range(0, NoOfSamples, BatchSize)]
    return MiniBatches

def GetActivation(ActivationType,X):

    return 

def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


    if (ActivationType=='CrossEntophy'):
        # E 
        print('CrossEntrophy')
        Activation=np.maximum(0,X)
        Activation=np.where(X < 0, 0, 1 )
        print(Activation)
    elif (ActivationType=='Binary'):
        # (x) = 1 if x > 0  else 0 if x < 0
        # The vector can contain xi that holds the value 0
        # and will be untouced. Only values differenct from 0
        # will be adjusted.
        # OK for binary classification
        print('Binary')
        Activation=np.maximum(0,X)
        Activation=np.where(X < 0, 0, 1 )
        print(Activation)
    elif (ActivationType=='ReLU'):
        # f(x) = max(0,X)
        print('ReLU')
        Activation=np.maximum(0,X)
        print(Activation)
    elif (ActivationType=='LReLU'):
        print('LReLU')
        Activation=np.where(X > 0, X, X * 0.01)
        print(Activation)
    elif (ActivationType=='Sigmoid'):
        # 𝜎(𝑥)= 1/(1+𝑒−𝑥)
        print('Sigmoid')
        Activation= (1/(1 + np.exp(-X)))
        print(Activation)
    elif (ActivationType=='SoftMax'):
        #SofMax usually used for output layers
        print('SoftMax')
        Expo = np.exp(X)
        ExpoSum = np.sum(np.exp(X))
        Activation=Expo/ExpoSum
        print(Activation)
    else:
        print('Else')


    def dlrelu(x, alpha=.01):
         # return alpha if x < 0 else 1

     return np.array ([1 if i >= 0 else alpha for i in x])
    return

def GetDerrivative (ActivationType,X):
    # We return the value of the derivative
    # Variable X = (WX+b)
    DerBackProp=0
    if (ActivationType=='Binary'):
        # (x) = 1 if x > 0  else 0 if x < 0
        # The vector can contain xi that holds the value 0
        # and will be untouced. Only values differenct from 0
        # will be adjusted.
        # OK for binary classification
        print('Binary')
        DerBackPro=np.where(X !=0,0,0.01)
        print(Activation)
    elif (ActivationType=='ReLU'):
        # f(x) = x[x<=0]=0 ; x[x>0] = 1
        print('ReLU')
        DerBackProp=np.array ([0 if i <= 0 else 1 for i in X])
        print(DerBackProp)
    elif (ActivationType=='LReLU'):
        print('LReLU')
        DerBackProp=np.array ([1 if i >= 0 else .01 for i in X])
        print(DerBackProp)
    elif (ActivationType=='Sigmoid'):
        # 𝜎(𝑥)= 1/(1+𝑒−𝑥)
        # 𝑑𝜎(𝑥)/𝑑(𝑥)= 𝜎(𝑥)⋅(1−𝜎(𝑥)).
        print('Sigmoid')
        DerBackProp= (1/(1 + np.exp(-X)))* (1 - (1/(1 + np.exp(-X))))
        print(DerBackProp)
    elif (ActivationType=='SoftMax'):
        # Softmax is usually used only for output layer
        # and should not be called, in this project.
        print('SoftMax')
        Expo = np.exp(X)
        ExpoSum = np.sum(np.exp(X))
        DerBackProp=Expo/ExpoSum
    else:
        print('Else')
    return DerBackProp

def GetAccuracyScore(YTest,YPredict,index):
    # Run the test data against the model and accumulate up
    # Accuracy = korrect (t=y) / number of tests
    # equivalent in numpy
    return np.sum(YTest == YPredict) / len(YTest)

def NNInitWeight(HLayers,Featurs):
    # Initialize weights by using Zavier balancing of weight
    # It takes into consideration the size of Ni. Input units of
    # the layer.
    # 1) Gaussian distribution, mean=1 std=1
    # 2) Multiply weights with 1/SQR(Ni)
    # -  W[l] = np.random.randn(l-1,l)*10 ---
    #
    # ensure the same random numbers appear every time
    np.random.seed(1776)
    
    W = [[]]
    for Layer in range(HLayers):
        W[Layer]=np.random.randn(Layer-1,Layer)*10

    return W

def NNInitBias(HiddenLayers):
    #The bias will counter balance the posibility of deminishin weight
    #and will be sett -> 1
    Size=range(HiddenLayers)
    Biases[:Size]=1
    
    return Biases

def NNUpdateWeights():
    return

def NNForwardPass():
    Epoch=1

    for i in range(Epoch):
        print('NNForwardPass')
    return

def NNBackWardPropagation():
    return

def NNMnist():
    # This FFNN is aimed towards linear regresion using
    # OLS, SGD, Sigmoid as the activation f'n for hidden layers and
    # SoftMax for the output layer.

    Epoch=1
    LearningRate= 0.01
    HiddenLayers=3
    Features=10
    #Set up struct for 

    # We need to create/decide the design matrix
    # 
    # download MNIST dataset
    MNistDigits = datasets.load_digits()

    inputs = MNistDigits.images
    Y = MNistDigits.target

    Weights=NNInitWeight(HiddenLayers,Features)
    Biases=NNInitBias(HiddenLayers)

    for i in range(Epoch):
        NNForwardPass()
        NNBackWardPropagation()
        NNUpdateWeights()
    ConfusionMatrix(YTest,Predictions)
    return

    

    return

def NN():
     ######## KRC #########

    # import necessary packages
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # ensure the same random numbers appear every time
    np.random.seed(1776)

    # display images in notebook
    #%matplotlib inline
    plt.rcParams['figure.figsize'] = (12,12)


    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
    print("labels = (n_inputs) = " + str(labels.shape))


    # flatten the image
    # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)
    print("X = (n_inputs, n_features) = " + str(inputs.shape))


    # choose some random images to display
    indices = np.arange(n_inputs)
    random_indices = np.random.choice(indices, size=5)

    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
    #plt.show()

    

    # equivalently in numpy
    def train_test_split_numpy(inputs, labels, train_size, test_size):
        n_inputs = len(inputs)
        inputs_shuffled = inputs.copy()
        labels_shuffled = labels.copy()
        
        np.random.shuffle(inputs_shuffled)
        np.random.shuffle(labels_shuffled)
        
        train_end = int(n_inputs*train_size)
        X_train, X_test = inputs_shuffled[:train_end], inputs_shuffled[train_end:]
        Y_train, Y_test = labels_shuffled[:train_end], labels_shuffled[train_end:]
        
        return X_train, X_test, Y_train, Y_test


    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def feed_forward(X):
        # weighted sum of inputs to the hidden layer
        z_h = np.matmul(X, hidden_weights) + hidden_bias
        # activation in the hidden layer
        a_h = sigmoid(z_h)
        
        # weighted sum of inputs to the output layer
        z_o = np.matmul(a_h, output_weights) + output_bias
        # softmax output
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        
        return probabilities

    
    # we obtain a prediction by taking the class with the highest likelihood
    def predict(X):
        probabilities = feed_forward(X)
        return np.argmax(probabilities, axis=1)

    # one-hot in numpy
    def to_categorical_numpy(integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1
        
        return onehot_vector


    def feed_forward_train(X):
        # weighted sum of inputs to the hidden layer
        z_h = np.matmul(X, hidden_weights) + hidden_bias
        # activation in the hidden layer
        a_h = sigmoid(z_h)
        
        # weighted sum of inputs to the output layer
        z_o = np.matmul(a_h, output_weights) + output_bias
        # softmax output
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        
        # for backpropagation need activations in hidden and output layers
        return a_h, probabilities

    def backpropagation(X, Y, a_h, probabilities):
        
        # error in the output layer
        error_output = probabilities - Y
        # error in the hidden layer
        error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
        
        # gradients for the output layer
        output_weights_gradient = np.matmul(a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)
        
        # gradient for the hidden layer
        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

    eta = 0.01
    lmbd = 0.01
    lamd = 1
    
    # one-liner from scikit-learn library
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                        test_size=test_size)
    
    X_train, X_test, Y_train, Y_test = train_test_split_numpy(inputs, labels, train_size, test_size)

    print("Number of training images: " + str(len(X_train)))
    print("Number of test images: " + str(len(X_test)))

    # building our neural network

    n_inputs, n_features = X_train.shape
    n_hidden_neurons = 50
    n_categories = 10

    # we make the weights normally distributed using numpy.random.randn

    # weights and bias in the hidden layer
    hidden_weights = np.random.randn(n_features, n_hidden_neurons)
    hidden_bias = np.zeros(n_hidden_neurons) + 0.01

    # weights and bias in the output layer
    output_weights = np.random.randn(n_hidden_neurons, n_categories)
    output_bias = np.zeros(n_categories) + 0.01

    # setup the feed-forward pass, subscript h = hidden layer

    probabilities = feed_forward(X_train)
    print("probabilities = (n_inputs, n_categories) = " + str(probabilities.shape))
    print("probability that image 0 is in category 0,1,2,...,9 = \n" + str(probabilities[0]))
    print("probabilities sum up to: " + str(probabilities[0].sum()))
    print()

    predictions = predict(X_train)
    print("predictions = (n_inputs) = " + str(predictions.shape))
    print("prediction for image 0: " + str(predictions[0]))
    print("correct label for image 0: " + str(Y_train[0]))

    # to categorical turns our integer vector into a onehot representation
    #from sklearn.metrics import accuracy_score
   
    #Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)
    print("Old accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))


    for i in range(100):
        # calculate gradients
        a_h, probabilities = feed_forward_train(X_train)
        dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train_onehot,a_h,probabilities)
        
        # regularization term gradients
        dWo += lmbd * output_weights
        dWh += lmbd * hidden_weights
        
        # update weights and biases
        output_weights -= eta * dWo
        output_bias -= eta * dBo
        hidden_weights -= eta * dWh
        hidden_bias -= eta * dBh
        print("New accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))

        ######## KRC #########

    return

def FFNNRegression():
    # This FFNN is aimed towards linear regresion using
    # OLS, SGD, Sigmoid as the activation f'n for hidden layers and
    # SoftMax for the output layer.

    Epoch=1
    LearningRate= 0.01
    HiddenLayers=3
    Features=5
    #Set up struct for 

    # We need to create/decide the design matrix
    # 
    Weights=NNInitWeight(HiddenLayers,Features)
    Biases=NNInitBias(HiddenLayers)

    for i in range(Epoch):
        NNForwardPass()
        NNBackWardPropagation()
        NNUpdateWeights()

    return

def MNISTLogisticRegression():
    
    # Trenger ikke dette
    N = 100
    D = 2
    X = np.random.randn(N,D)

    #Read data from MNIST or use a global
    #varialbe for the other rutines
    #GetMNISTData()
    MNistDigits = datasets.load_digits()
    inputs = MNistDigits.images
    T = MNistDigits.target


    #Trenger ikke dette
    # center the first 50 points at (-2,-2)
    X[:50,:] = X[:50,:] - 2*np.ones((50,D))

    #Trenger ikke dette heller
    # center the last 50 points at (2, 2)
    X[50:,:] = X[50:,:] + 2*np.ones((50,D))

    # Trenger ikke dette heller
    # labels: first 50 are 0, last 50 are 1
    T = np.array([0]*50 + [1]*50)

    #Trenger ikke dette, men må sette opp en DesignMatrix
    # add a column of ones
    # ones = np.array([[1]*N]).T
    ones = np.ones((N, 1))
    Xb = np.concatenate((ones, X), axis=1)

    #Fyll ut dette
    #N=AntallTraingSet
    #X=DesignMatrix(MNIST)
    #ones = np.ones((N, 1))
    #Xb = np.concatenate((ones, X), axis=1)

    # randomly initialize the weights
    # Expect the D=64 (8 x 8)
    w = np.random.randn(D + 1)

    # calculate the model output
    z = Xb.dot(w)

    def sigmoid(z):
        return 1/(1 + np.exp(-z))


    Y = sigmoid(z)

    # calculate the cross-entropy error
    def cross_entropy(T, Y):
        E = 0
        for i in range(len(T)):
            if T[i] == 1:
                E -= np.log(Y[i])
            else:
                E -= np.log(1 - Y[i])
        return E


    # let's do gradient descent 100 times
    learning_rate = 0.001
    for i in range(1000000):
        if i % 10 == 0:
            print(cross_entropy(T, Y))

        # gradient descent weight udpate with regularization
        # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.1*w ) # old
        w += learning_rate * ( Xb.T.dot(T - Y) - 0.001*w )

        # recalculate Y
        Y = sigmoid(Xb.dot(w))


    print("Final w:", w)
    return

def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}").format(
                    j, self.evaluate(test_data), n_test)
            else:
                print("Epoch {0} complete").format(j)

def OLSRegression(X, y):
    # Runt the linear gression using MSE and 
    # returnt the Ypredict
    #
    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100


    theta = np.random.randn(2,1)  # random initialization

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    
    #To be deleted
    Yhat=[]
    ###### KRC ###
    # perform gradient descent to find w
    costs = [] # keep track of squared error cost
    w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
    learning_rate = 0.001
    l1 = 10.0 # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
    for t in range(500):
    # update w
        Yhat = X.dot(w)
        delta = Yhat - Y
        w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    # find and store the cost
    mse = delta.dot(delta) / N
    costs.append(mse)

    # plot the costs
    plt.plot(costs)
    plt.show()

    print("final w:", w)

    # plot our w vs true w
    plt.plot(true_w, label='true w')
    plt.plot(w, label='w_map')
    plt.legend()
    plt.show()
    return

def TestSGDLinearReg():
    # In this routing we will test the SGD aginst the
    # OLS and Ridge.
    # We use the classical sample...
    Samples=100
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    plt.xlabel("Lamdas") 
    pltTitle=str(Samples) + " Samples"
    pltFigName=CurrentPath + "/" + "-Samples-" + str(Samples) + "Points-"+".png"
    #pltFigName=CurrentPath + "/" + str(ReggrType)+ "-Bootstrap" + "-Samples-" + str(Samples) + "Lamdas-"+".png"

    # Dette skal byttes med tall fra OLS
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    y_predict

    #plt.ylabel("Loss/Error",fontsize=18)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    

    plt.title(pltTitle)
    plt.axis([0, 2, 0, 15])
    plt.plot(X_new, y_predict, "r-", label="Pred Normal")
    plt.plot(X, y, "b.", label="points")
    #plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()
    return

def ReadDataLogRegression():
    print('Read inn data')
    return

def DesignLogRegressionModel():
    print('Design model')
    return

# Part 2-A
# Write the part of the code which reads in the data and sets up the relevant data sets.
#
def LogisticRegression():
    TestRegression()
    # ReadDataLogRegression()
    # DesignLogRegressionModel()
    return

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
        #ThreadLock.release()
        send_end.send('self.LockId')
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

#from concurrent.futures import ThreadPoolExecutor
#import requests


#from timer import timer

#def ThreadPool():
#    with ThreadPoolExecutor(max_workers=10) as executor
#        start=time.perf_counter()
#        finished=time.perf_counter()
#       print(f'Finised in {round(finished-start,2)} second(s)')
#   return

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
          
    def __init__(self, Architecture,InitBias, InitWeight,ThreadID, Name, LockId,Flag,X_train,y_train,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end):
        self.Architecture = Architecture
        self.Bias = InitBias
        self.InitWeight = InitWeight
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
        #ThreadLock.release()
        self.send_end.send(self.LockId)
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
    
    Epochs=20000   
    
    Processes=[]
    pipe_list = []
    #for i in range(NumberOfProcesses):
    for i in range(1):
        recv_end, send_end = Pipe(False)
        InitBias=InitBias+0.02
        Name=Name+str(i)
        #Epochs +=2000
        #LRate=LRate+0.001
        List=[NN_ARCHITECTURE,InitBias,InitWeight,i,Name,i,Flag, X_train, y_train,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end]
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
    print('All processes finished!')
            
    # Loop on all nets
    # ThreadNet0 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-0", 0,2, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet0.SetThreadId(ThreadNet0)
    # ThreadNet0.start()
    # ThreadNet1 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-1", 1,3, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet1.SetThreadId(ThreadNet1)
    # ThreadNet1.start()
    # ThreadNet2 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-2", 2,4, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet2.SetThreadId(ThreadNet2)
    # ThreadNet2.start()
    # ThreadNet4 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-3", 3,5, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet4.SetThreadId(ThreadNet4)
    # ThreadNet4.start()
    # ThreadNet5 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-4", 4,6, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet5.SetThreadId(ThreadNet5)
    # ThreadNet5.start()
    # ThreadNet6 = DeepNeuralNettworkMultiThreaded(NN_ARCHITECTURE,0.01,1,1,"Thread-5", 5,1, X_train, y_train,Epochs,LRate,True,True)
    # ThreadNet6.SetThreadId(ThreadNet6)
    # ThreadNet6.start()
    
    #Threads.append(ThreadNet1)

    #for t in Threads:
    #    t.join()
    #    print ("Exiting Main Thread")
    #print('All treads finished!')

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
    
    Epochs=20000   
    
    Threads=[]
    pipe_list = []
    for i in range(NumberOfProcesses):
        recv_end, send_end = Pipe(False)
        InitBias=InitBias+0.02
        Name=Name+str(i)
        #Epochs +=2000
        #LRate=LRate+0.001
        List=[NN_ARCHITECTURE,InitBias,InitWeight,i,Name,i,Flag, X_train, y_train,Epochs,LRate,Verbose,ShowLoss,SyncLoss,send_end]
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
    #TestCases3DNeuralNetwork(1,0.003)
    #TestCases3DNeuralNetwork(2,0.003)
    #TestCases3DNeuralNetwork(3,0.003)
    #TestCases3DNeuralNetwork(1,0.003,3001)
    #TestCases3DNeuralNetwork(2,0.003,3001)
    #TestCases3DNeuralNetwork(3,0.003,3001)

    #TestCases3DNeuralNetwork(3,0.001,3001)

    #TestCases3DNeuralNetwork(4,0.001,5001)

    ####### BEST FIT ###############3
    #TestCases3DNeuralNetwork(5,0.0003,8001)
    #TestCases3DNeuralNetwork(6,0.0003,10001)
    #TestCases3DNeuralNetwork(6,0.0001,30001)

    #######   TEST 3-D FFNN  ###############
    #   We start with a 10x10x3 
    #TestCases3DNeuralNetwork(1,0.0003,3001)
    #TestMultiThreadedNerworks(1,0.0003,8001,1)
    #TestOneTreadOfNetwork(1,0.0003,8001,1)

    #TestCases3DNeuralNetwork(6,0.0003,10001)
    #TestCases3DNeuralNetwork(6,0.0001,30001)
    
    
    #Testing processes with different "meta" parameters
    TestMultiThreadedNerworks(1,0.0003,8001,1)
    #TestParallelProcessOfNetwork(1,0.0003,8001,1)
    
  


def Main():
    #This module is central for running an testing both Project 1 & 2
    #
    Samples = 2000
    x = np.random.uniform(0,1,size=Samples)
    y = np.random.uniform(0,1,size=Samples)

    #Scale the data based on subracting the mean
    x=SubMean(x)
    y=SubMean(y)

    #Part-A
    # Test for plynomials up to 5'th degree by using the Franke's function
    # Regression analysis using OLS/MSE, to find confidence intervals,
    # varances, MSE. Use scaling of the data (subtractiong mean) and add noise.
    #
    # Test ut saving a files with .png extension
    global CurrentPath
    CurrentPath=MakeDirAndReturnPath("/Plots")

    PolyDim=20
    z = FrankeFunction(x, y)
    z = AddNoise(z,Samples)
    X = DesignMatrix(x,y,PolyDim)
    Lamdas = [0.0001, 0.001, 0.01, 0.1, 1, 2]
    Bootstraps=1000
    Folds=5
    
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