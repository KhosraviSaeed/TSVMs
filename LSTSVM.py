"""
Article : Least squares twin support vector machines for pattern classification
Link    : https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/S0957417408006854
Author  : Saeed Khosravi
"""
import numpy as np
class LSTSVM:
    """
        Least Squares Support Vector Machines
        A = Instances with label +1
        B = Instances with label -1
        C1 = hyperparameter for hyperplane 1
        C2 = hyperparameter for hyperplane 2
        
    """
    def __init__(self, X, y, C1, C2, eps = 1e-4):
        self.A = X[np.ix_(y[:,0] ==   1),:][0,:,:]
        self.B = X[np.ix_(y[:,0] ==  -1),:][0,:,:]
        self.C1 = C1
        self.C2 = C2
        self.eps = eps
        
    def fit(self):
        A = self.A
        B = self.B
        C1 = self.C1
        C2 = self.C2
        eps = self.eps
        m1, n = A.shape
        m2, n = B.shape
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))
        X = np.concatenate((A, B), axis=0)
        G = np.concatenate((A, e1), axis=1)
        H = np.concatenate((B, e2), axis=1)
        

        if(m1 < m2):
            Y = self.calc_Y_or_Z(H)
           
            #w1, b1
            GYGT = np.dot(np.dot(G, Y), G.T)
            I = np.eye(GYGT.shape[0], GYGT.shape[1])
            w1_b1 = - np.dot(Y - np.dot(np.dot(np.dot(Y, G.T), np.linalg.inv(C1*I + GYGT)), np.dot(G, Y)), 
                             np.dot(H.T, np.ones((H.T.shape[1], 1))))
            w1 = w1_b1[:-1,  :]
            b1 = w1_b1[ -1,  :]
            
            #w2, b2
            w2_b2 = C2 * np.dot(Y - np.dot(np.dot(np.dot(Y, G.T), np.linalg.inv((I/C2)+GYGT)), np.dot(G, Y)), 
                                np.dot(G.T, np.ones((G.T.shape[1], 1))))
            w2 = w2_b2[:-1,  :]
            b2 = w2_b2[ -1,  :]
            
        else:
            Z = self.calc_Y_or_Z(G)
            
            #w1, b1
            HZHT = np.dot(np.dot(H, Z), H.T)
            I = np.eye(HZHT.shape[0], HZHT.shape[1])
            w1_b1 = -C1*np.dot(Z - np.dot(np.dot(np.dot(Z, H.T), np.linalg.inv((I/C1) + HZHT)), np.dot(H, Z)),
                               np.dot(H.T, np.ones((H.T.shape[1], 1))))
            w1 = w1_b1[:-1,  :]
            b1 = w1_b1[ -1,  :]
            
            #w2, b2
            w2_b2 = np.dot(Z - np.dot(np.dot(np.dot(Z, H.T), np.linalg.inv(C2*I + HZHT)), np.dot(H, Z)), 
                           np.dot(G.T, np.ones((G.T.shape[1], 1))))
            w2 = w2_b2[:-1,  :]
            b2 = w2_b2[ -1,  :]

        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    
    def predict(self, x_test, y_test):
        distance1 = np.abs(np.dot(x_test, self.w1) + self.b1)
        distance2 = np.abs(np.dot(x_test, self.w2) + self.b2)
        y_pred = np.zeros_like(y_test)
        for d in range(y_pred.shape[0]):
            if (distance1[d] < distance2[d]):
                y_pred[d][0] =  1;
            else:
                y_pred[d][0] = -1;
        self.preds = y_pred
    
    def calc_Y_or_Z(self, M):
        MMT = np.dot(M, M.T)
        I = np.eye(MMT.shape[0], MMT.shape[1])
        tmp = np.dot(np.dot(M.T, np.linalg.inv(self.eps*I + MMT)), M)
        I = np.eye(tmp.shape[0], tmp.shape[1])
        return (1/self.eps)*(I-tmp)
    
    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    

    def get_preds(self):
        return self.preds