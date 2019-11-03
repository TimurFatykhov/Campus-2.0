import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def get_X_y():
    """
    Загружает данные с цифрами
    
    
    Возвращает две переменные:
    -------------------------
    - X: матрица 1797x64
        картинки в виде векторов длинной 64
        
    - y: матрица 1797x10
        матрица, где в каждой строке только в одном столбце единица,
        о в остальных столбцах нули: единица стоит в столбце,
        который соотетствует цифре из матрицы X, то есть если 
        y[m] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], то X[m] - картинка
        с нулем, если y[k] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        то X[k] - картинка с девяткой и тд
        
    
    Пример использования:
    ---------------------
    >>> X, y = get_X_y()
    >>>
    """
    X, y_raw = load_digits(return_X_y=True)
    n = len(X)
    y = np.zeros((n, 10))
    y[range(0,n), y_raw] = 1
    
    return X, y


def show_image(img, figsize=(5,5)):
    """
    Показывает изображение
    
    Параметры:
    - img: numpy.array
        массив numpy, с тремя или одним каналом (цветное или ч/б фото)
    """
    if len(img.shape) < 2:
        s = np.sqrt(len(img)).astype(int)
        img = img.reshape((s,s))
        
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
def ReLU(vector):
    v_copy = vector.copy()
    
    mask = v_copy < 0
    v_copy[mask] = 0
    
    return v_copy


def softmax(s):
    e = np.exp(s - s.max(1).reshape((-1, 1)) )
    summ = np.sum(e)
    return e / summ


def mean_square_error(predict, true):
    N = len(predict)
    return np.sum((predict - true)**2) / N

    
class TwoLayerClassifier():
    
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.rand(input_size, hidden_size) * 2 - 1
        self.params['b1'] = np.random.rand(hidden_size)
        
        self.params['W2'] = np.random.rand(hidden_size, output_size) * 2 - 1
        self.params['b2'] = np.random.rand(output_size)
        
        self.H = None
        self.S = None
        self.P = None
    
    
    def loss(self, X, y=None, reg=0):
        N = len(X)
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        # Compute the forward pass
        self.o_11 = X.dot(W1)
        self.o_12 = self.o_11 + b1
        self.H = ReLU(self.o_12)
        
        self.o_21 = self.H.dot(W2)
        self.S = self.H.dot(W2) + b2
        
        self.P = softmax(self.S)
        
        if y is None:
            return self.P
        
        loss = mean_square_error(self.P, y)
        
        # Backward pass: compute gradients 
        grads = {}
        
        # dl/dP
        # dP
        dP = 2 * (self.P - y)
        
        # dl/dP * dP/dS
        # dS
        exp = np.exp(self.S - self.S.max(1).reshape((-1, 1)))
        softsum = np.sum(exp, 1).reshape((-1, 1))
        dS = exp * (softsum - exp) / softsum**2
        dS = dP * dS
        
        # dl/dP * dP/dS * dS/dW2
        # dW2
        grads['W2'] = self.H.T.dot(dS) / N
        
        # dl/dP * dP/dS * dS/db2
        # db2
        grads['b2'] = dS.sum(0) / N
        
        # dH
        dH = dS.dot(W2.T)
        
        # do_12
        do_12 = np.ones_like(self.o_12)
        do_12[self.o_12 < 0] = 0
        
        # dW1
        grads['W1'] = X.T.dot(do_12) / N
        
        # db1
        grads['b1'] = do_12.sum(0) / N
        
        return loss, grads
        
        
    def fit(self, X, y, epochs, lr):
        self.history_loss = []
        
        for epoch in range(epochs):
            loss, grads = self.loss(X, y)
            # print(loss)
            self.history_loss.append(loss)
            
            self.params['W1'] -= lr * grads['W1']
            self.params['b1'] -= lr * grads['b1']
            self.params['W2'] -= lr * grads['W2']
            self.params['b2'] -= lr * grads['b2']
        
        

    










