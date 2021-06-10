import numpy as np

class OLS():

	"""Modelo linear dos Mínimos Quadrados Ordinários estilo sklearn, com métodos fit e predict"""
	def __init__(self):

		self.w = None

	def fit(self, X, y):
		#criando o vetor bias:
		X = np.c_[np.ones(X.shape[0]), X]

		parametro = np.linalg.inv(X.T @ X) @ X.T @ y
		self.w = parametro

	def predict(self, X):
		#bias
		X = np.c_[np.ones(X.shape[0]), X]

		return X @ self.w


class RegressaoGD():
	"""Modelo linear de Regressão com gradiente descendente estilo sklearn,
	   com métodos fit e predict, armazena os custos e os parâmetros referentes a cada época"""
	def __init__(self, t=1000, taxa = 0.01):
		self.w = None
		self.t = t
		self.taxa = taxa
		self.w_passados = []
		self.custos = []

	def fit(self, X, y):
		X = np.c_[np.ones(X.shape[0]), X]
		self.w = np.random.rand(X.shape[1])

		for epoca in range(self.t):
			
			pred = X @ self.w
			erro = y - pred

			custo = (1/(2*len(y))) * np.sum((y - pred)**2)
			
			self.w = self.w + (self.taxa * (1/len(y)) * (X.T @ erro))
			# Armazenando o histórico
			self.custos.append(custo)
			self.w_passados.append(self.w)

	def predict(self, X):
		X = np.c_[np.ones(X.shape[0]), X]
		return X @ self.w


class SGD():

	"""Modelo linear de Regressão com gradiente descendente estocásticos estilo sklearn,
	   com métodos fit e predict, armazena os custos e os parâmetros referentes a cada iteração"""

    def __init__(self, t=1000, taxa = 0.01):

        self.w = None
        self.t = t
        self.taxa = taxa
        self.w_passados = []
        self.custos = []

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]

        self.w = np.random.rand(X.shape[1])

        for epoca in range(self.t):
            indice_aleatorio = np.random.permutation(len(y))

            for xi, yi in zip(X[indice_aleatorio], y[indice_aleatorio]):

                pred = xi @ self.w
                erro = yi - pred
                self.w = self.w + self.taxa * erro * xi

                custo = (1/(2*len(y))) * np.sum((y - X @ self.w)**2)
                self.custos.append(custo)
                self.w_passados.append(self.w)



    def predict(self, X):
    	X = np.c_[np.ones(X.shape[0]), X]
    	return X @ self.w

class RegressaoPolinomial():
	"""Modelo de Regressão Polinomial com gradiente descendente estilo sklearn,
	   com métodos fit e predict, esse modelo cria atrbutos polinomais que
	   podem se ajustar a não linearidades nos atributos originais, 
	   Regularização L2 inclusa."""
	   
	def __init__(self, t=1000, taxa = 0.01, lamb = 0, ordem = 1):
		self.w = None
		self.t = t
		self.taxa = taxa
		self.w_passados = []
		self.custos = []
		self.lamb = lamb
		self.ordem = ordem

	def fit(self, X, y):

		if self.ordem == 1:
			X = np.c_[np.ones(X.shape[0]), X]

		if self.ordem > 1:
			for coluna in range(X.shape[1]):
				for i in range(2,self.ordem+1):
					X = np.c_[X, X[:, coluna]**i]

			X = np.c_[np.ones(X.shape[0]), X]

		self.w = np.random.rand(X.shape[1])

		for epoca in range(self.t):
			#X = 30,2 , w = (2,) X @ w = (30,)
			pred = X @ self.w
			erro = y - pred

			custo = (1/(2*len(y))) * np.sum((y - pred)**2)
			
			termo = self.lamb*self.w
			termo[0] = 0

			self.w = self.w + (self.taxa * ((1/len(y)) * (X.T @ erro) - termo))
			# Armazenando o histórico
			self.custos.append(custo)
			self.w_passados.append(self.w)

	def predict(self, X):

		if self.ordem == 1:
			X = np.c_[np.ones(X.shape[0]), X]

		if self.ordem > 1:
			for coluna in range(X.shape[1]):
				for i in range(2,self.ordem+1):
					X = np.c_[X, X[:, coluna]**i]

			X = np.c_[np.ones(X.shape[0]), X]

			
		return X @ self.w









