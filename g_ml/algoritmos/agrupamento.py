import numpy as np

#centroides = D x K
#distancias = n x K
def Kmeans(K, X, n_init):
    D = X.shape[1]
    n = len(X)
    tol = 0.001
    data = np.c_[X, np.zeros(n)]

    erro_por_init = []
    centroides_por_init = []
    historico_por_init = []
    historico_C = []

    for i in range(n_init):
        #inicialização
        erros_hist = []
        centroides = np.zeros((D,K))
        for i in range(K):
            i_aleatorio=np.random.randint(0,len(X))
            centroides[:, i] = X[i_aleatorio]

        erro_anterior = 0
        dif = 999
        while dif > tol:
            #parte 1            
            euclidistancias = np.zeros((n,K))
            for k in range(K):
                distancia_k = np.sqrt(np.sum((X-centroides[:,k])**2,axis=1))
                euclidistancias[:,k]=distancia_k

            C=np.argmin(euclidistancias,axis=1)
            data[:, -1] = C

            grupos  = np.unique(data[:, -1])
            #parte 2
            erro_recon = 0
            for grupo in grupos:
                X_grupo = data[:,0:D][data[:, -1] == grupo]            
                m_grupo = np.sum(X_grupo, axis = 0)/len(X_grupo)
                centroides[:, int(grupo)] = m_grupo
                erro_recon = erro_recon + np.sum(np.power((X_grupo-centroides[:, int(grupo)]),2))

            dif = np.abs(erro_recon - erro_anterior)
            #print(dif)
            erros_hist.append(erro_recon)
            erro_anterior = erro_recon

        erro_por_init.append(np.min(erros_hist))
        historico_por_init.append(erros_hist)
        centroides_por_init.append(centroides)

        melhor_centroide = centroides_por_init[np.argmin(erro_por_init)]
        melhor_historico = historico_por_init[np.argmin(erro_por_init)]
        menor_erro = np.min(erro_por_init)

        euclidistancias = np.zeros((n,K))
        for k in range(K):
            distancia_k = np.sqrt(np.sum((X-melhor_centroide[:,k])**2,axis=1))
            euclidistancias[:,k]=distancia_k

        C=np.argmin(euclidistancias,axis=1)
        historico_C.append(C)
        melhor_C = historico_C[np.argmin(erro_por_init)]
    
    return melhor_centroide, melhor_historico, melhor_C, menor_erro

class PCA():
    def __init__(self):
        self.cov = None
        self.valores = None
        self.vetores = None
        self.var_explicada = None
    
    def fit(self, X):
        self.cov = np.cov(X.T)
        self.valores, self.vetores = np.linalg.eig(self.cov)
        self.var_explicada = self.valores/np.sum(self.valores)
    
    def transform(self, X, m):
        self.P = self.vetores[0:m]
        self.u = np.mean(X, axis=0)
        self.z = (X - self.u) @ self.P.T
        return self.z
    
    def reconstruir(self):
        return self.z @ self.P + self.u