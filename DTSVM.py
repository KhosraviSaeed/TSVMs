"""
Article : Deep Twin Support Vector Machine
Link    : https://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/7022580
Author  : Saeed Khosravi

"""

import numpy as np
import TSVM

class DTSVM:
    
    def __init__(self, X, y, C, eps = 1e-4):
        
        self.X   = X
        self.y   = y
        self.C   = C
        self.eps = eps
        
    def fit(self):
        #LSTSVM 1
        C1 = self.C[0]
        C2 = self.C[1]
        y  = self.y
        tsvm1 = TSVM.TSVM(self.X, y, C1, C2)
        tsvm1.fit()
        self.w11, self.b11, self.w12, self.b12 = tsvm1.get_params()
        self.f1 = self.f_(self.X, self.w11, self.b11, self.w12, self.b12)

        #LSTSVM 2
        C1 = self.C[2]
        C2 = self.C[3]
        y  = self.y
        tsvm2 = TSVM.TSVM(self.X, y, C1, C2)
        tsvm2.fit()
        self.w21, self.b21, self.w22, self.b22 = tsvm2.get_params()
        self.f2 = self.f_(self.X, self.w21, self.b21, self.w22, self.b22)

        #LSTSVM Main
        C1 = self.C[4]
        C2 = self.C[5]
        X = self.f(self.f1, self.f2)
        y = self.y
        tsvm_M = TSVM.TSVM(X, y, C1, C2)
        tsvm_M.fit()
        self.w1, self.b1, self.w2, self.b2 = tsvm_M.get_params()
        

    def predict(self, x_test, y_test):
        f1 = self.f_(x_test, self.w11, self.b11, self.w12, self.b12)
        f2 = self.f_(x_test, self.w21, self.b21, self.w22, self.b22)
        f = self.f(f1, f2)
        distance_1 = np.abs(np.dot(f, self.w1) + self.b1)
        distance_2 = np.abs(np.dot(f, self.w2) + self.b2)
        y_pred = np.zeros_like(distance_1)
        for i in range(y_pred.shape[0]):
            if (distance_1[i] < distance_2[i]):
                y_pred[i][0] =  1;
            else:
                y_pred[i][0] = -1;
        self.preds = y_pred
 
    def f_(self, x, w1, b1, w2, b2):
        f  = np.concatenate((np.dot(x, w1)+b1, np.dot(x, w2)+b2), axis = 1)
        return f
        
    def f(self, f1, f2):
        f = np.concatenate((f1,f2), axis = 1)
        return f
    
    
    def get_hidden_params(self):
        return self.w11, self.b11, self.w12, self.b12, self.w21, self.b21, self.w22, self.b22
    
    def get_output_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds