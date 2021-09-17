"""
Article : Twin Support Vector Machine
Link    : https://sci-hub.tw/https://ieeexplore.ieee.org/document/4135685
Author  : Saeed Khosravi
"""

import numpy as np
from cvxopt import solvers, matrix

class TSVM:
    
    def __init__(self, X, y, C1, C2, eps=1e-4):
        
        self.A   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        self.B   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        self.C1  = C1
        self.C2  = C2
        self.eps = eps
        
    def fit(self):
        self.w1, self.b1 = self.plane1(self.A, self.B, self.C1, self.eps)
        self.w2, self.b2 = self.plane2(self.A, self.B, self.C2, self.eps)
        
    def predict(self, x_test):
        norm2_w1 = np.linalg.norm(self.w1)
        norm2_w2 = np.linalg.norm(self.w2)
        distance_1 = np.abs(np.dot(x_test, self.w1) + self.b1)/norm2_w1
        distance_2 = np.abs(np.dot(x_test, self.w2) + self.b2)/norm2_w2
        y_pred = np.zeros_like(distance_1)
        for i in range(y_pred.shape[0]):
            if (distance_1[i] < distance_2[i]):
                y_pred[i][0] =  1;
            else:
                y_pred[i][0] = -1;

        self.preds = y_pred
        
    def plane1(self, A, B, c, eps):
        e1  = np.ones((A.shape[0],1))
        e2  = np.ones((B.shape[0],1))
        H   = np.concatenate((A,e1), axis=1)
        G   = np.concatenate((B,e2), axis=1)
        HTH = np.dot(H.T, H)
        if np.linalg.matrix_rank(H)<H.shape[1]:
            HTH += eps*np.eye(HTH.shape[0], HTH.shape[1])
        
        _P = matrix(np.dot(np.dot(G, np.linalg.inv(HTH)),G.T), tc = 'd')
        _q = matrix(-1 * e2, tc = 'd')
        _G = matrix(np.concatenate((np.identity(B.shape[0]),-np.identity(B.shape[0])), axis=0), tc = 'd')
        _h = matrix(np.concatenate((c*e2,np.zeros_like(e2)), axis=0), tc = 'd')
        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        z = -np.dot(np.dot(np.linalg.inv(HTH), G.T), qp_sol)
        w = z[:z.shape[0]-1]
        b = z[z.shape[0]-1]
        return w, b[0]
    
    def plane2(self, A, B, c, eps):
        e1  = np.ones((A.shape[0],1))
        e2  = np.ones((B.shape[0],1))
        H   = np.concatenate((A,e1), axis=1)
        G   = np.concatenate((B,e2), axis=1)
        GTG = np.dot(G.T, G)
        if np.linalg.matrix_rank(G)<G.shape[1]:
            GTG += eps*np.eye(GTG.shape[0], GTG.shape[1])
        #solving the qp by cvxopt
        _P = matrix(np.dot(np.dot(H, np.linalg.inv(GTG)), H.T), tc = 'd')
        _q = matrix(-1 * e1, tc = 'd')
        _G = matrix(np.concatenate((np.identity(A.shape[0]),-np.identity(A.shape[0])), axis=0), tc = 'd')
        _h = matrix(np.concatenate((c*e1,np.zeros_like(e1)), axis=0), tc = 'd')
        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        z = -np.dot(np.dot(np.linalg.inv(GTG), H.T), qp_sol)
        w = z[:z.shape[0]-1]
        b = z[z.shape[0]-1]
        return w, b[0]
    
    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds