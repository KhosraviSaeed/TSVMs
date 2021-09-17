"""
Article : A reduced universum twin support vector machine for class imbalance learning
Link    : https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/S0031320319304510
Author  : Saeed Khosravi
"""

import numpy as np
from cvxopt import solvers, matrix
import math

class RUTSVM:
    
    def __init__(self, X, y, C1, C2, CU, eps):
        
        self.X   = X
        self.y   = y
        self.C1  = C1
        self.C2  = C2
        self.CU  = CU
        self.eps = eps
        
    def fit(self):
        self.w1, self.b1 = self.plane1(self.X, self.y, self.C1, self.CU, self.eps)
        self.w2, self.b2 = self.plane2(self.X, self.y, self.C2, self.CU, self.eps)
        
    def predict(self, x_test):
        norm2_w1 = np.linalg.norm(self.w1)
        norm2_w2 = np.linalg.norm(self.w2)
        distance_1 = np.abs(np.dot(x_test, self.w1) + self.b1)/norm2_w1
        distance_2 = np.abs(np.dot(x_test, self.w2) + self.b2)/norm2_w2
        y_pred = np.zeros_like(distance_1).reshape((-1, 1))
        for i in range(y_pred.shape[0]):
            if (distance_1[i] < distance_2[i]):
                y_pred[i][0] =  1;
            else:
                y_pred[i][0] = -1;

        self.preds = y_pred
        
    def plane1(self, X, y, c1, cu, eps):
        S, T, O, T_, O_, e1, e2, eg, ed = self.split_dataset(X, y)
        m1 = S.shape[0]
        m2 = T_.shape[0]
        mg = O_.shape[0]

        STS = np.dot(S.T, S)
        I = np.eye(STS.shape[0], STS.shape[1])
        STS_inv = np.linalg.inv(1e-4*I + STS)

        _P = np.dot(np.dot(T_, STS_inv), T_.T)
        _P = np.concatenate((_P, -np.dot(np.dot(T_, STS_inv), O_.T)), axis = 1)
        _P2 = -np.dot(-np.dot(O_, STS_inv), T_.T)
        _P2 = np.concatenate((_P2, np.dot(np.dot(O_, STS_inv), O_.T)), axis = 1)
        _P = np.concatenate((_P, _P2), axis = 0) # (m1 + mg , m1 + mg)

        _q = np.concatenate((-e1.T, (1-eps)*eg.T), axis = 1).T # (m1 + mg , 1)

        _G1 = np.concatenate(( np.eye(m1, m1), np.zeros((m1, mg))), axis = 1)
        _G2 = np.concatenate((-np.eye(m1, m1), np.zeros((m1, mg))), axis = 1)
        _G3 = np.concatenate(( np.zeros((mg, m1)), np.eye(mg, mg)), axis = 1)
        _G4 = np.concatenate((np.zeros((mg, m1)), -np.eye(mg, mg)), axis = 1)
        _G  = np.concatenate((_G1, _G2), axis = 0)
        _G  = np.concatenate((_G , _G3), axis = 0)
        _G  = np.concatenate((_G, _G4), axis = 0) # (2m1 + 2mg , m1 + mg)

        _h = np.zeros((2*m1 + 2*mg, 1))
        _h[:m1, :] = c1
        _h[2*m1:2*m1+mg, :] = cu

        _P = matrix(_P, tc= 'd')
        _q = matrix(_q, tc = 'd')
        _G = matrix(_G, tc = 'd')
        _h = matrix(_h, tc = 'd')

        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        alphas = qp_sol[:m1, 0]
        mus    = qp_sol[m1:, 0]
        vp  = -np.dot(STS_inv, np.dot(T_.T, alphas) - np.dot(O_.T, mus))
        w = vp[:-1]
        b = vp[-1]
        return w, b
    
    
    def plane2(self, X, y, c2, cu, eps):
        S, T, O, T_, O_, e1, e2, eg, ed = self.split_dataset(X, y)

        m1 = S.shape[0]
        m2 = T.shape[0]
        md = O.shape[0]

        TTT = np.dot(T.T, T)
        I = np.eye(TTT.shape[0], TTT.shape[1])
        TTT_inv = np.linalg.inv(1e-4*I + TTT)
        _P = np.dot(np.dot(S, TTT_inv), S.T)
        _P = np.concatenate((_P, -np.dot(np.dot(S, TTT_inv), O.T)), axis = 1)
        _P2 = -np.dot(-np.dot(O, TTT_inv), S.T)
        _P2 = np.concatenate((_P2, np.dot(np.dot(O, TTT_inv), O.T)), axis = 1)
        _P = np.concatenate((_P, _P2), axis = 0) # (m1 + md , m1 + md)

        _q = np.concatenate((-e1.T, (eps - 1)*ed.T), axis = 1).T # (m1 + md , 1)

        _G1 = np.concatenate(( np.eye(m1, m1), np.zeros((m1, md))), axis = 1)
        _G2 = np.concatenate((-np.eye(m1, m1), np.zeros((m1, md))), axis = 1)
        _G3 = np.concatenate(( np.zeros((md, m1)), np.eye(md, md)), axis = 1)
        _G4 = np.concatenate((np.zeros((md, m1)), -np.eye(md, md)), axis = 1)
        _G  = np.concatenate((_G1, _G2), axis = 0)
        _G  = np.concatenate((_G , _G3), axis = 0)
        _G  = np.concatenate((_G, _G4), axis = 0) # (2m1 + 2md , m1 + md)

        _h = np.zeros((2*m1 + 2*md, 1))
        _h[:m1, :] = c2
        _h[2*m1:2*m1+md, :] = cu


        _P = matrix(_P, tc= 'd')
        _q = matrix(_q, tc = 'd')
        _G = matrix(_G, tc = 'd')
        _h = matrix(_h, tc = 'd')

        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        alphas = qp_sol[:m1, 0]
        mus    = qp_sol[m1:, 0]
        vn  = -np.dot(TTT_inv, np.dot(S.T, alphas) - np.dot(O.T, mus))
        w = vn[:-1]
        b = vn[-1]
        return w, b
        
    def split_dataset(self, X, y):
        X1   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        X2   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        r, n = X1.shape
        s, n = X2.shape
        X2_  = X2[:r, :]
        U    = X2[r: , :]
        d, n = U.shape
        g    = math.ceil(r/2)
        tmp  = np.random.choice(np.arange(1, d), g)
        U_   = U[tmp, :]
        e1 = np.ones((X1.shape[0], 1))
        e2 = np.ones((X2.shape[0], 1))
        eg = np.ones((U_.shape[0], 1))
        ed = np.ones((U.shape[0] , 1))
        S  = np.concatenate((X1 , e1), axis = 1)
        T_ = np.concatenate((X2_, e1), axis = 1)
        T  = np.concatenate((X2 , e2), axis = 1)
        O_ = np.concatenate((U_ , eg), axis = 1)
        O  = np.concatenate((U  , ed), axis = 1)
        return S, T, O, T_, O_, e1, e2, eg, ed
    
    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds