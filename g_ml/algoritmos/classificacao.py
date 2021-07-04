import numpy as np

class RegressaoLogistica():

    def __init__(self, t=1000, taxa=0.005):
        self.w = None
        self.t = t
        self.taxa = 0.01
        self.w_passados = []
        self.custos = []

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.random.rand(X.shape[1]) * 0.9

        for epoca in range(self.t):
            # Predição convencional
            pred = X @ self.w
            # Predição usando função logística
            y_pred = 1/(1+np.exp(-pred))
            
            # calculando o erro
            erro = y - y_pred
            #calculando o custo
            custo =  np.mean(-y * np.log(y_pred) - (1-y) * np.log(1 - y_pred))
            
            #atualizando os parâmetros
            
            self.w = self.w +  self.taxa * (X.T @ erro)/len(y)

            # Armazenando o histórico
            self.custos.append(custo)
            self.w_passados.append(self.w)

    def predict(self, X):

        X = np.c_[np.ones(X.shape[0]), X]
        y_pred = 1.0/(1+np.exp(-X @ self.w))
        return np.where( y_pred > 0.5, 1, 0 )

class ADG():

    def __init__(self, proporcional = True):

        self.classes = None
        self.n_classes = None
        self.n_dim = None
        self.prioris = None
        self.medias = None
        self.covs = None
        self.proporcional = proporcional

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_dim = X.shape[1]

        if self.proporcional == True:
            self.prioris = []
            for classe in self.classes:
                self.prioris.append((y==classe).sum()/len(y))
            self.prioris = np.array(self.prioris)
        else:
            self.prioris = np.ones(n_classes) * 1/n_classes
   
        self.medias = np.zeros((self.n_dim, self.n_classes))
        self.covs = np.zeros((self.n_dim, self.n_dim, self.n_classes))

        #finalmente atualizando os parâmetros:
        for classe in self.classes:

            X_classe = X[y==classe]

            self.medias[:, classe] = np.mean(X_classe, axis=0)
            self.covs[:, :, classe] = (X_classe - self.medias[:, classe]).T @ (X_classe - self.medias[:, classe]) / (len(X_classe) - 1)
    def predict(self, X):
        y_pred = []
        for linha in range(len(X)):
            prob_classe = []
            for classe in range(self.n_classes):
                prob_classe.append( np.log(self.prioris[classe]) - 0.5 * np.log(np.linalg.det(self.covs[:,:,classe])) - 0.5 * (X[linha, :] - self.medias[:, classe]).T @ np.linalg.inv(self.covs[:,:,classe]) @ (X[linha,:] - self.medias[:,classe]))
            y_pred.append(self.classes[np.argmax(prob_classe)])
        y_pred = np.array(y_pred)
        return y_pred

class NaiveBayesGaussiano():

    def __init__(self, proporcional = True):

        self.classes = None
        self.n_classes = None
        self.n_dim = None
        self.prioris = None
        self.medias = None
        self.vars = None
        self.proporcional = proporcional

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_dim = X.shape[1]

        if self.proporcional == True:
            self.prioris = []
            for classe in self.classes:
                self.prioris.append((y==classe).sum()/len(y))
            self.prioris = np.array(self.prioris)
        else:
            self.prioris = np.ones(n_classes) * 1/n_classes
   
        self.medias = np.zeros((self.n_dim, self.n_classes))
        self.vars = np.zeros((self.n_dim, self.n_classes))

        #finalmente atualizando os parâmetros:
        for classe in self.classes:

            X_classe = X[y==classe]

            self.medias[:, classe] = np.mean(X_classe, axis=0)
            self.vars[:, classe] = np.sum((X_classe - self.medias[:, classe])**2, axis=0) / (len(X_classe) - 1)

    def predict(self, X):
        y_pred = []
        for linha in range(len(X)):
            prob_classe = []
            for classe in range(self.n_classes):
                prob_classe.append( np.log(self.prioris[classe]) - 0.5 * np.sum(np.log(2 * np.pi * self.vars[:, classe])) - 0.5* np.sum((X[linha, :] - self.medias[:, classe])**2 / self.vars[:, classe]))
            y_pred.append(self.classes[np.argmax(prob_classe)])
        y_pred = np.array(y_pred)
        return y_pred

class KNN():
    
    def __init__(self):
        self.K = None
        self.X_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X, K=1):
        self.K = K
        classes = np.unique(self.y_train)
        y_pred = []
        
        for xi in X:
            euclidistancias = -2 * xi @ self.X_train.T + (xi**2).sum() + (self.X_train**2).sum(axis=1)
            knn_index = np.argsort(euclidistancias)[0:K]
            contagem = []
            for classe in classes:
                contagem.append((self.y_train[knn_index]==classe).sum())

            contagem = np.array(contagem)
            pred_index = np.argmax(contagem)    
            y_pred.append(classes[pred_index])

        return np.array(y_pred)
