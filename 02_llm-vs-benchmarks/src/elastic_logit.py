import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy.sparse

class ElasticNetLogit(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Elastic Net (L1 + L2) regularization.
    """
    def __init__(
        self, beta=1.0, gamma=1.0, learning_rate=1e-3, max_iter=200, 
        batch_size=1024, device=None, random_state=42
    ):
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.loss_history = []

        self.device = device

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        num_features = X.shape[1]
        
        self.model = nn.Linear(num_features, num_classes, dtype=torch.float32, device=self.device)
        
        is_sparse = scipy.sparse.issparse(X)
        if not is_sparse:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            
        y_mapped = np.searchsorted(self.classes_, y)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long, device=self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.beta)
        criterion = nn.CrossEntropyLoss()
        
        num_samples = X.shape[0]
        self.loss_history =[]
        
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            permutation = torch.randperm(num_samples, device=self.device)
            
            for i in range(0, num_samples, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                
                if is_sparse:
                    batch_X_np = X[indices.cpu().numpy()].toarray()
                    batch_X = torch.tensor(batch_X_np, dtype=torch.float32, device=self.device)
                else:
                    batch_X = X[indices]
                    
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if self.gamma > 0:
                    l1_loss = sum(torch.linalg.norm(p, 1) for p in self.model.parameters())
                    loss += self.gamma * l1_loss
                    
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.shape[0]
                
            self.loss_history.append(epoch_loss / num_samples)
            
        self.w = self.model.weight.detach().cpu().numpy().T
        return self

    def predict_proba(self, X):
        is_sparse = scipy.sparse.issparse(X)
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                if is_sparse:
                    batch_X_np = X[i:i + self.batch_size].toarray()
                    batch_X = torch.tensor(batch_X_np, dtype=torch.float32, device=self.device)
                else:
                    batch_X = torch.tensor(X[i:i + self.batch_size], dtype=torch.float32, device=self.device)
                    
                outputs = self.model(batch_X)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                
        return np.vstack(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]