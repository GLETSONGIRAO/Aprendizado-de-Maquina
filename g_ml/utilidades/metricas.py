import numpy as np

def MSE(true, pred):
	"""Função para calcular o erro quadrático médio entre valores reais e valores preditos"""

	return np.mean((true - pred)**2)

def MRE(y_true ,y_pred):
	"""Função para calcular o erro relativo médio entre valores reais e valores preditos"""

	return np.mean(np.abs((y_true - y_pred)/y_true))
