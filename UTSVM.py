"""
Article : Twin Support Vector Machine with Universum data
Link    : https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/S0893608012002304
Author  : Saeed Khosravi
"""

import numpy as np
from cvxopt import solvers, matrix

class UTSVM:
    
    def __init__(self, X, y, U, C1, C2, CU, eps):
        
        self.X   = X
        self.y   = y
        self.U   = U
        self.C1  = C1
        self.C2  = C2
        self.CU  = CU
        self.eps = eps
        
    def fit(self):
        self.w1, self.b1 = self.plane1(self.X, self.y, self.U, self.C1, self.CU, self.eps)
        self.w2, self.b2 = self.plane2(self.X, self.y, self.U, self.C2, self.CU, self.eps)
        
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
        
    def plane1(self, X, y, U, c, cu, eps):
        A   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        B   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        m1 = A.shape[0]
        m2 = B.shape[0]
        ep = np.ones((m1, 1))
        en = np.ones((m2, 1))
        mu = U.shape[0]
        eu = np.ones((mu, 1))
        H = np.concatenate((A, ep), axis = 1)
        G = np.concatenate((B, en), axis = 1)
        O = np.concatenate((U, eu), axis = 1)
        HTH = np.dot(H.T, H)
        I = np.eye(HTH.shape[0], HTH.shape[1])
        HTH_inv = np.linalg.inv(1e-4*I + HTH)

        _P = np.dot(np.dot(G, HTH_inv), G.T)
        _P = np.concatenate((_P, -np.dot(np.dot(G, HTH_inv), O.T)), axis = 1)
        _P2 = -np.dot(np.dot(O, HTH_inv), G.T)
        _P2 = np.concatenate((_P2, np.dot(np.dot(O, HTH_inv), O.T)), axis = 1)
        _P = np.concatenate((_P, _P2), axis = 0) # (en + eu , en + eu)


        _q = np.concatenate((-en.T, (1-eps)*eu.T), axis = 1).T # (en + eu , 1)


        _G1 = np.concatenate(( np.eye(m2, m2), np.zeros((m2, mu))), axis = 1)
        _G2 = np.concatenate((-np.eye(m2, m2), np.zeros((m2, mu))), axis = 1)
        _G3 = np.concatenate(( np.zeros((mu, m2)), np.eye(mu, mu)), axis = 1)
        _G4 = np.concatenate(( np.zeros((mu, m2)), -np.eye(mu, mu)), axis = 1)
        _G = np.concatenate((_G1, _G2), axis = 0)
        _G = np.concatenate((_G , _G3), axis = 0)
        _G = np.concatenate((_G, _G4), axis = 0) # (4 * m2, m2 + mu)

        _h = np.zeros((2*m2 + 2*mu, 1))
        _h[:m2, 0] = c
        _h[2*m2:2*m2+mu, :] = cu

        _P = matrix(_P, tc= 'd')
        _q = matrix(_q, tc = 'd')
        _G = matrix(_G, tc = 'd')
        _h = matrix(_h, tc = 'd')
        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        alphas = qp_sol[:m2, 0]
        mus    = qp_sol[m2:, 0]
        vp  = -np.dot(HTH_inv, np.dot(G.T, alphas) - np.dot(O.T, mus))
        w = vp[:vp.shape[0]-1]
        b = vp[vp.shape[0]-1]
        return w, b
    
    def plane2(self, X, y, U, c, cu, eps):
        A   = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        B   = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        m1 = A.shape[0]
        m2 = B.shape[0]
        en = np.ones((m1, 1))
        ep = np.ones((m2, 1))
        mu = U.shape[0]
        eu = np.ones((mu, 1))
        Q = np.concatenate((A, en), axis = 1)
        P = np.concatenate((B, ep), axis = 1)
        S = np.concatenate((U, eu), axis = 1)
        QTQ = np.dot(Q.T, Q)
        I = np.eye(QTQ.shape[0], QTQ.shape[1])
        QTQ_inv = np.linalg.inv(1e-4*I + QTQ)

        _P = np.dot(np.dot(P, QTQ_inv), P.T)
        _P = np.concatenate((_P, -np.dot(np.dot(P, QTQ_inv), S.T)), axis = 1)
        _P2 = -np.dot(np.dot(S, QTQ_inv), P.T)
        _P2 = np.concatenate((_P2, np.dot(np.dot(S, QTQ_inv), S.T)), axis = 1)
        _P = np.concatenate((_P, _P2), axis = 0) # (ep + eu , ep + eu)


        _q = np.concatenate((-ep.T, (1-eps)*eu.T), axis = 1).T # (ep + eu , 1)


        _G1 = np.concatenate(( np.eye(m2, m2), np.zeros((m2, mu))), axis = 1)
        _G2 = np.concatenate((-np.eye(m2, m2), np.zeros((m2, mu))), axis = 1)
        _G3 = np.concatenate(( np.zeros((mu, m2)), np.eye(mu, mu)), axis = 1)
        _G4 = np.concatenate(( np.zeros((mu, m2)), -np.eye(mu, mu)), axis = 1)
        _G = np.concatenate((_G1, _G2), axis = 0)
        _G = np.concatenate((_G , _G3), axis = 0)
        _G = np.concatenate((_G, _G4), axis = 0) # (4 * m2, m2 + mu)

        _h = np.zeros((2*m2 + 2*mu, 1))
        _h[:m2, 0] = c
        _h[2*m2:2*m2+mu, :] = cu

        _P = matrix(_P, tc= 'd')
        _q = matrix(_q, tc = 'd')
        _G = matrix(_G, tc = 'd')
        _h = matrix(_h, tc = 'd')
        qp_sol = solvers.qp(_P, _q, _G, _h, kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})
        qp_sol = np.array(qp_sol['x'])
        alphas = qp_sol[:m2, 0]
        mus    = qp_sol[m2:, 0]
        vp  = -np.dot(QTQ_inv, np.dot(P.T, alphas) - np.dot(S.T, mus))
        w = vp[:vp.shape[0]-1]
        b = vp[vp.shape[0]-1]
        return w, b

    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds