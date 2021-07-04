import numpy as np
import matplotlib.pyplot as plt

#------------------------> Regressão <----------------------------#

def MSE(true, pred):
	"""Função para calcular o erro quadrático médio entre valores reais e valores preditos, para tarefas de regressão."""

	return np.mean((true - pred)**2)

def MRE(y_true ,y_pred):
	"""Função para calcular o erro relativo médio entre valores reais e valores preditos, para tarefas de regressão."""

	return np.mean(np.abs((y_true - y_pred)/y_true))

#------------------------> Classificação <----------------------------#

def ACC(y_verdadeiro, y_pred):
	"""Função para calcular dentre todas as classificações, quantas o modelo classificou corretamente.
	Fórmula: (VP + VN) / (VP + VN + FP + FN)"""

	return (y_verdadeiro == y_pred).sum()/len(y_pred)

class relatorio_classificacao():
	"""
	Fontes: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
			https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
	--------------------------------------------------------------------------
	   Descrição: Classe para calcular e mostrar as principais métricas de Classificação, 
	   			  Recebe y_verdadeiro e y_predito na inialização.
	--------------------------------------------------------------------------
	   Atributos:
	   
	   Acurácia = self.acc 
	   precisão = self.precisao 
	   revocação = self. revocacao
	   f1_score = self.f1_score)
	----------------------------------------------------------------------------
	   Métodos:

	   print_scores(): Printa na tela as métricas calculadas.
	   plot_scores(): Mostra gráfico com a matriz de confusão.


	"""

	def __init__(self, y_pred, y_teste):

		self.pos = (y_pred == True).sum()
		self.neg = (y_pred == False).sum() 

		self.vp = y_teste[y_pred== 1].sum()
		self.fp = self.pos - y_teste[y_pred== 1].sum()

		self.fn = y_teste[y_pred== False].sum()
		self.vn = self.neg - y_teste[y_pred== False].sum()
        
		self.acc = (self.vp + self.vn) / (self.vp + self.vn + self.fp + self.fn)
		self.precisao = self.vp / (self.vp + self.fp)
		self.revocacao = self.vp / (self.vp + self.fn)
		self.f1_score = (2 * self.precisao * self.revocacao) / (self.precisao + self.revocacao)

	def print_scores(self):		

		print("A acurácia do modelo é de: {} ".format(self.acc))
		print("A precisão do modelo é de: {}".format(self.precisao))
		print("A revocacão do modelo é de: {}".format(self.revocacao))
		print("O f1_score do modelo é de: {}".format(self.f1_score))
        
	def plot_scores(self):
		array = np.array([[self.vn, self.fp],[self.fn, self.vp]])
		fig, ax = plt.subplots(figsize = (12,7))
		im = ax.imshow(array,cmap='gray')
		ax.set_yticks([])
		ax.set_xticks([])
		ax.set_title('Matriz de Confusão')
		text = ax.text(0, 0, array[0, 0],fontsize='x-large',fontweight='bold',ha="center", va="top", color="#00ff00")
		text = ax.text(0, 0, "Verdadeiro Negativo",fontweight='bold',fontsize = 'x-large',ha="center", va="bottom", color="#00ff00")

		text = ax.text(0, 1, array[0, 1],fontsize='x-large',fontweight='bold',ha="center", va="top", color="r")
		text = ax.text(0, 1, "Falso Positivo",fontweight='bold',fontsize = 'x-large',ha="center", va="bottom", color="r")

		text = ax.text(1, 0, array[1, 0],fontsize='x-large',fontweight='bold',ha="center", va="top", color="r")
		text = ax.text(1, 0, "Falso Negativo",fontweight='bold',fontsize = 'x-large',ha="center", va="bottom", color="r")

		text = ax.text(1, 1, array[1, 1],fontsize='x-large',fontweight='bold',ha="center", va="top", color="#00ff00")
		text = ax.text(1, 1, "Verdadeiro Positivo",fontweight='bold',fontsize = 'x-large',ha="center", va="bottom", color="#00ff00")

#Validação cruzada adapatada apenas para classificação por enquanto.	
class ValCruzada():
    
    def __init__(self, X ,part=10):
        # Ao inicializar ele salva os índices das partições aleatórias dos dados
        # Para que sejam usados as mesmas entradas para diferentes modelos e técnicas de pré-processamento
        
        self.part = part
        self.indice_aleatorio = np.random.permutation(len(X))
        
        self.X_folds = None
        self.y_folds = None
        
        self.X_testes = None
        self.X_treinos = None
        self.y_testes = None
        self.y_treinos = None
        
        self.preds = None
        self.acc = None
        self.precisao = None
        self.revocacao = None
        self.f1_score = None
          
    def particionar(self, X, y):
        # Esse método separa os dados em seus devidos conjuntos de treino e teste para cada uma das partições. 
        
        self.X_folds = []
        self.y_folds = []
        
        self.X_testes = []
        self.X_treinos = []
        self.y_testes = []
        self.y_treinos = []
        
        _X = X[self.indice_aleatorio].copy()
        _y = y[self.indice_aleatorio].copy()
        index = np.floor(np.linspace(0,len(_X), self.part+1)).astype('int')
        
        for i in range(self.part):
            start = index[i]
            end = index[i+1]
            self.X_folds.append(_X[start:end,:])
            self.y_folds.append(_y[start:end])
            
        for i in range(len(self.X_folds)):
            _X_folds = self.X_folds.copy()
            _y_folds = self.y_folds.copy()
            
            self.X_testes.append(_X_folds.pop(i))
            self.X_treinos.append(np.concatenate(_X_folds,axis=0))
            
            self.y_testes.append(_y_folds.pop(i))
            self.y_treinos.append(np.concatenate(_y_folds))
            
            
            
    def medir(self, modelo):
        # Calcula as predições, as métricas e estatísticas para cada partição.
        
        self.preds = []
        self.acc = []
        self.precisao = []
        self.revocacao = []
        self.f1_score = []
        
        for i in range(self.part):
            modelo.fit(self.X_treinos[i], self.y_treinos[i])
            self.preds.append(modelo.predict(self.X_testes[i]))
        
        for i in range(self.part):
            metricas = relatorio_classificacao(self.preds[i], self.y_testes[i])
            self.acc.append(metricas.acc)
            self.precisao.append(metricas.precisao)
            self.revocacao.append(metricas.revocacao)
            self.f1_score.append(metricas.f1_score)
            
        print("Acurácia média é: {} e seu desvio padrão é: {} ".format(np.mean(self.acc),np.std(self.acc)))
        print("Precisão média é: {} e seu desvio padrão é: {} ".format(np.mean(self.precisao),np.std(self.precisao)))
        print("Revocacão média é: {} e seu desvio padrão é: {} ".format(np.mean(self.revocacao),np.std(self.revocacao)))
        print("F1_score médio é: {} e seu desvio padrão é: {} ".format(np.mean(self.f1_score),np.std(self.f1_score)))            
        
          
