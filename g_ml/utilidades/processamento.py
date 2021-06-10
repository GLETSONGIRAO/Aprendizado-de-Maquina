import numpy as np

def treino_teste(X, y, tamanho_teste = 0.33):
	"""Função parar separar atributos e alvo de entrada em treino e teste, recebe X, y e a proporção e retorna X e y de treino e X e y de teste"""

	indice_aleatorio = np.random.permutation(len(y))
	limiar = int(len(y) * tamanho_teste)
	X_teste = X[indice_aleatorio][0:limiar]
	X_treino = X[indice_aleatorio][limiar:]
	y_teste = y[indice_aleatorio][0:limiar]
	y_treino = y[indice_aleatorio][limiar:]
	return X_treino, X_teste, y_treino, y_teste

class escala_padrao():

	def __init__(self):

		self.media = None
		self.dp = None

	def fit(self, dados):

		self.media = dados.mean(axis=0)
		self.dp = dados.std(axis=0)

	def transform(self, dados):

		return (dados - self.media) / self.dp

	def inversa(self, dados):

		return (dados * self.dp) + self.media

class escala_min_max():
	def __init__(self):
		self.max = None
		self.min = None
	def fit(self, dados):
		self.max = dados.max()
		self.min = dados.min()
	def transform(self, dados):
		return (dados - self.min)/ (self.max-self.min)
	def inversa(self, dados):
		return dados * (self.max - self.min) + self.min



	

    
    
    
    
    
    

