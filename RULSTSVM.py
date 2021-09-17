"""
Article: Reduced Universum Least Squares Support Vector Machine
Link   : New
Author : Saeed Khosravi
"""

import numpy as np
import math

class RULSTSVM:
    def __init__(self, X, y, C, eps):
        self.X   = X
        self.y   = y
        self.C   = C
        self.eps = eps
    
    def fit(self):
        self.plane1(self.X, self.y, self.C[0], self.C[1], self.C[2], self.eps)
        self.plane2(self.X, self.y, self.C[3], self.C[4], self.C[5], self.eps)
    
    def predict(self, x_test):
        distance_1 = np.abs(np.dot(x_test, self.w1) + self.b1)
        distance_2 = np.abs(np.dot(x_test, self.w2) + self.b2)
        y_pred = np.zeros_like(distance_1).reshape((-1, 1))
        for i in range(y_pred.shape[0]):
            if (distance_1[i] < distance_2[i]):
                y_pred[i][0] =  1;
            else:
                y_pred[i][0] = -1;
        self.preds = y_pred
        
    def plane1(self, X, y, C1, C2, C3, eps):
        S, T_, O_, e1, eg = self.definitions1(X, y)
        STS   = np.dot(S.T, S)
        T_TT_ = np.dot(T_.T, T_)
        O_TO_ = np.dot(O_.T, O_)
        I     = np.eye(STS.shape[0], STS.shape[1])
        v1    = -np.dot(np.linalg.inv(STS + C1*T_TT_ + C2*I + C3*O_TO_), np.dot(C1*T_.T, e1) + (1-eps)*C3*np.dot(O_.T, eg))
        self.w1    = v1[:-1, :]
        self.b1    = v1[ -1, :]  
    
    def plane2(self, X, y, C4, C5, C6, eps):
        S, T, O, e1, ed = self.definitions2(X, y)
        TTT = np.dot(T.T, T)
        STS = np.dot(S.T, S)
        OTO = np.dot(O.T, O)
        I   = np.dot(TTT.shape[0], TTT.shape[0])
        v2  = np.dot(np.linalg.inv(TTT + C4*STS + C5*I + C6*OTO), C4*np.dot(S.T, e1) - C6*np.dot(O.T, (1-eps)*ed))
        self.w2    = v2[:-1, :]
        self.b2    = v2[ -1, :]
        
    def definitions1(self, X, y):
        X1   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        X2   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        r, n = X1.shape
        s, n = X2.shape
        np.random.shuffle(X2)
        X2_  = X2[:r,  :]
        U    = X2[r: , :]
        d, n = U.shape
        g    = math.ceil(r/2)
        U_   = U[np.random.choice(np.arange(1, d), g), :]
        e1   = np.ones((X1.shape[0], 1))
        eg   = np.ones((U_.shape[0], 1))
        S    = np.concatenate((X1 , e1), axis = 1)
        T_   = np.concatenate((X2_, e1), axis = 1)
        O_   = np.concatenate((U_ , eg), axis = 1)
        return S, T_, O_, e1, eg

    def definitions2(self, X, y):
        X1   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        X2   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        r, n = X1.shape
        s, n = X2.shape
        np.random.shuffle(X2)
        X2_  = X2[:r,  :]
        U    = X2[r: , :]
        d, n = U.shape
        g    = math.ceil(r/2)
        e1   = np.ones((X1.shape[0], 1))
        e2   = np.ones((X2.shape[0], 1))
        ed   = np.ones((U.shape[0] , 1))
        S    = np.concatenate((X1 , e1), axis = 1)
        T    = np.concatenate((X2 , e2), axis = 1)
        O    = np.concatenate((U  , ed), axis = 1)
        return S, T, O, e1, ed
    
    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds