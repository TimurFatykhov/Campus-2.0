import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class NN():
    def __init__(self, model, criterion, lr=1e-3, optimizer=None):
        """
        If optimizer is passed, then lr will be ignored
        """
        self.model = model
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
        self.train_history = []
        self.valid_history = []
        
        # self.criterion = torch.nn.NLLLoss()
        self.criterion = criterion
        
        
    def save(self, path='gesture_classifier.pt'):
        torch.save(self.model.to('cpu'), path)
        print('Model has been saved at "%s"' % (os.path.join(os.getcwd(), path)))
    
    
    def loss(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        self.model.eval()
        loss = None
        
        with torch.no_grad():
            outputs = self.forward(X, batch_size)
            outputs = torch.FloatTensor(outputs).to(self.device)
            y = torch.LongTensor(y).to(self.device)

            loss = self.criterion(outputs, y).item()
            
        return loss
    
    
    def show_history(self, hide_left=0):
        if self.valid_history is not None:
            N = len(self.train_history)
            plt.plot(np.arange(hide_left, N), self.train_history[hide_left:], color='blue', label='train')
        else: 
            print('Сначала обучите нейросеть!')
            return

        if self.valid_history is not None:
            plt.plot(np.arange(hide_left, N), self.valid_history[hide_left:], color='green', label='val')

        plt.legend()
        plt.grid()
        plt.show()
        
        
    def forward(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - batch_size: int
        """
        self.model.to(self.device)
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        N = len(X)
        
        masks = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                X_batch = X[i : min(i + batch_size, N)]

                masks.append(self.model(X_batch))
            
        masks = torch.cat(masks).to('cpu')
        return masks
    
    
    def fit(self, X, y, epochs, batch_size, valid_data=None, log_every_epoch=None):
        """
        Parameters:
        -----------
        - X: numpy.array
        
        - y: numpy.array
        
        - batch_size: int
        
        - valid_data: tuple (numpy.array, numpy.array)
            (X_valid, y_valid)
            
        - log_every_epoch: int
        """
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)

        N = len(X)
        
        bar = tqdm(range(1, epochs+1)) # progress bar
        for epoch in bar:
            cum_loss_train = 0
            part = 0
            for i in range(0, N, batch_size):
                self.model.train()
                
                part += 1
                X_batch = X[i : min(i + batch_size, N)]
                y_batch = y[i : min(i + batch_size, N)]

                outputs = self.model(X_batch)

                loss = self.criterion(outputs, y_batch)
                
                cum_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.train_history.append(cum_loss_train / part)
                
            if valid_data is not None:
                valid_loss = self.loss(valid_data[0], valid_data[1], batch_size)
                self.valid_history.append(valid_loss)
                    
            if log_every_epoch is not None and epoch % log_every_epoch == 0:
                descr = None
                t_loss = self.train_history[-1]
                descr = ('t_loss: %5.3f' % t_loss)
                
                if valid_data is not None:
                    v_loss = self.valid_history[-1]
                    descr += ('v_loss: %5.3f' % v_loss)
                    
                bar.set_description(descr)
    
    
    def fit_loader(self, train_loader, epochs, valid_data=None, valid_batch_size=128, log_every_epoch=None):
        self.model.to(self.device)
        
        bar = tqdm(range(1, epochs+1)) # progress bar
        for epoch in bar:
            cum_loss_train = 0
            part = 0
            for x, y in train_loader:
                self.model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                loss = self.criterion(outputs, y)
                
                cum_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.train_history.append(cum_loss_train / len(train_loader))
                
                
            if valid_data is not None:
                valid_loss = self.loss(valid_data[0], valid_data[1], valid_batch_size)
                self.valid_history.append(valid_loss)
                    
            if log_every_epoch is not None and epoch % log_every_epoch == 0:
                descr = None
                t_loss = self.train_history[-1]
                descr = ('t_loss: %5.3f' % t_loss)
                
                if valid_data is not None:
                    v_loss = self.valid_history[-1]
                    descr += ('v_loss: %5.3f' % v_loss)
                    
                bar.set_description(descr)
        
    
class NNSegmentation(NN):
    def show_predict_grid(self, X, y, size=5, figsize=(15, 15), threshold=0.1):
        pred = self.predict(X[:size*size])
        pred = (pred > threshold).astype(float)
        
        fig, ax = plt.subplots(size, size, figsize=figsize)
        ax = np.ravel(ax)
        
        for i, arr in enumerate(X[:size*size, 0].copy()):
            mask = arr + pred[i, 1] * 100
            mask = np.clip(mask, 0, 1)
            
            arr[pred[i, 1] > 0] *= 0.35
            
            image = np.transpose(np.stack([arr, arr, mask]), (1, 2, 0))
            # mask = np.transpose(np.stack([pred[i, 1], np.zeros_like(pred[i, 1]), np.zeros_like(pred[i, 1])]), (1, 2, 0))
            
            ax[i].imshow(image)
            ax[i].axis('off')
            # ax[i].set_title('%d (%d)' % (pred[i], y[i]), color=color)
        
        plt.show()
    
    def predict(self, X, batch_size=128):
        out = self.forward(X, batch_size)
        return F.softmax(out, dim=1).numpy()
        

class NNClassifier(NN):
    def predict_proba(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        N = len(X)
        
        proba = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                X_batch = X[i : min(i + batch_size, N)]

                proba.append(self.model(X_batch))
            
        proba = torch.cat(proba).to('cpu').numpy()
        return proba
    
    
    def predict(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        proba = self.predict_proba(X, batch_size)
        predict = proba.argmax(1)
        return predict
    
    
    def evaluate_score(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        predict = self.predict(X, batch_size)
        return (predict == y).mean()
        
    
    def show_predict_grid(self, X, y, size=5, figsize=(15, 15)):
        pred = self.predict(X)
        
        fig, ax = plt.subplots(size, size, figsize=figsize)
        ax = np.ravel(ax)
        
        for i, img in enumerate(X[:size*size]):
            color = 'green' if y[i] == pred[i] else 'red'
            
            ax[i].imshow(np.transpose(X[i], (1, 2, 0)))
            ax[i].axis('off')
            ax[i].set_title('%d (%d)' % (pred[i], y[i]), color=color)
        
        plt.show()
                    
                    
class Flatten(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)

    
class Softmax_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        e = torch.exp(x - x.max(1, True)[0] )
        summ = e.sum(1, True)[0]
        return e / summ