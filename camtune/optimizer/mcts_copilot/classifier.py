import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List

from botorch.utils.transforms import normalize, unnormalize, standardize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import silhouette_score

KMEANS_PARAMS = {
    'n_clusters': 2,
    'n_init': 'auto',
}
DBSCAN_PARAMS = {
    'eps': 0.1,
    'min_samples': 2,
}
CLUSTER_PARAMS = {
    'kmeans': {
        'cls': KMeans,
        'params': KMEANS_PARAMS,
    },
    'dbscan': {
        'cls': DBSCAN,
        'params': DBSCAN_PARAMS,
    },
}

CLASSIFIER_PARAMS = {
    'svm': {
        'cls': SVC,
        'params': {
            'kernel': 'rbf',
            'gamma': 'auto',
        }
    },
    'dt': {
        'cls': DTC,
        'params': {}
    }
}

class BaseClassifier:
    def __init__(
            self, 
            bounds: torch.Tensor, 
            seed: int, 
            classifier_type: str,
            classifier_params: Dict,
            cluster_type: str,
            cluster_params: Dict,
            fit_feat_vals: bool=True,
    ) -> None:
        self.bounds = bounds
        self.seed = seed
        self.fit_feat_vals = fit_feat_vals

        self.init_classifier(classifier_type, classifier_params)
        self.init_cluster(cluster_type, cluster_params)
        self.cluster_score: float = None
    
    def init_classifier(self, classifier_type: str, classifier_params: Dict):
        self.classifier_type = classifier_type
        if classifier_type not in CLASSIFIER_PARAMS:
            raise ValueError(f"[BaseClassifier] Invalid classifier type: {classifier_type}")
        
        default_params = CLASSIFIER_PARAMS[classifier_type]['params']
        for key, value in default_params.items():
            if key not in classifier_params:
                classifier_params[key] = value
        self.classifier_params = classifier_params

        self.classifier_cls = CLASSIFIER_PARAMS[classifier_type]['cls']
        self.classifier = self.classifier_cls(random_state=self.seed, **self.classifier_params)
        
    
    def init_cluster(self, cluster_type: str, cluster_params: Dict):
        self.cluster_type = cluster_type
        if cluster_type not in CLUSTER_PARAMS:
            raise ValueError(f"[BaseClassifier] Invalid cluster type: {cluster_type}")
        
        default_params = CLUSTER_PARAMS[cluster_type]['params']
        for key, value in default_params.items():
            if key not in cluster_params:
                cluster_params[key] = value
        self.cluster_params = cluster_params

        self.cluster_cls = CLUSTER_PARAMS[cluster_type]['cls']
        if cluster_type == 'dbscan':
            self.clusterizer = self.cluster_cls(**self.cluster_params)
        else:
            self.clusterizer = self.cluster_cls(random_state=self.seed, **self.cluster_params)
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> bool:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        # Because the data essentially with association to the performance is the unnormalized ones,
        # we need to denormalize the data before feeding into the KMeans and SVM
        
        # X_scaled - (num_samples, dimension)        
        X_scaled = X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # labels: ndarray with shape (num_samples, )
        labels, cluster_data = self.cluster(X_scaled, Y)
        if len(set(labels)) == 1:
            return False
    
        self.cluster_score = silhouette_score(cluster_data, labels)
        self.classifier = self.classifier.fit(X_scaled.detach().cpu().numpy(), labels)
        return True
    
    def cluster(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """

        # X_and_Y - (num_samples, dimension + 1)
        if self.fit_feat_vals:
            X_and_Y = torch.cat((X, Y), dim=1)
            cluster_data = X_and_Y.detach().cpu().numpy()
        else:
            cluster_data = Y.detach().cpu().numpy()

        labels = self.clusterizer.fit_predict(cluster_data)
        
        return labels, cluster_data

    def predict(self, X: torch.Tensor) -> np.array:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)

        Returns:
            labels: ndarray with shape (num_samples, )
        """
        X_scaled = X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return self.classifier.predict(X_scaled.detach().cpu().numpy())

    def save(self, save_path: str):
        import pickle
        with open(save_path,'wb') as f:
            pickle.dump(self.classifier, f)

class StandardClassifier(BaseClassifier):
    def __init__(self, bounds: torch.Tensor, seed: int, classifier_type: str, classifier_params: Dict, cluster_type: str, cluster_params: Dict, fit_feat_vals: bool = True) -> None:
        super().__init__(bounds, seed, classifier_type, classifier_params, cluster_type, cluster_params, fit_feat_vals)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> bool:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        std_Y = standardize(Y)
        
        # labels: ndarray with shape (num_samples, )
        labels, cluster_data = self.cluster(X, std_Y)
        if len(set(labels)) == 1:
            return False
    
        self.cluster_score = silhouette_score(cluster_data, labels)
        self.classifier = self.classifier.fit(X.detach().cpu().numpy(), labels)
        return True
    
    def cluster(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        # Now the input X for clustering is normalized (within the range of [0, 1])
        if self.fit_feat_vals:
            X_and_Y = torch.cat((X, Y), dim=1)
            cluster_data = X_and_Y.detach().cpu().numpy()
        else:
            cluster_data = Y.detach().cpu().numpy()

        labels = self.clusterizer.fit_predict(cluster_data)
        return labels, cluster_data

    def predict(self, X: torch.Tensor) -> np.array:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)

        Returns:
            labels: ndarray with shape (num_samples, )
        """
        # Now the input X for clustering is normalized (within the range of [0, 1])
        return self.classifier.predict(X.detach().cpu().numpy())
    
class StandardClassifierII(BaseClassifier):
    def __init__(self, bounds: torch.Tensor, seed: int, classifier_type: str, classifier_params: Dict, cluster_type: str, cluster_params: Dict, fit_feat_vals: bool = True) -> None:
        super().__init__(bounds, seed, classifier_type, classifier_params, cluster_type, cluster_params, fit_feat_vals)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> bool:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        # labels: ndarray with shape (num_samples, )
        labels, cluster_data = self.cluster(X, Y)
        if len(set(labels)) == 1:
            return False
    
        self.cluster_score = silhouette_score(cluster_data, labels)
        self.classifier = self.classifier.fit(X.detach().cpu().numpy(), labels)
        return True
    
    def cluster(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        # Now the input X for clustering is normalized (within the range of [0, 1])
        if self.fit_feat_vals:
            X_and_Y = torch.cat((X, Y), dim=1)
            cluster_data = X_and_Y.detach().cpu().numpy()
        else:
            cluster_data = Y.detach().cpu().numpy()

        labels = self.clusterizer.fit_predict(cluster_data)
        return labels, cluster_data

    def predict(self, X: torch.Tensor) -> np.array:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)

        Returns:
            labels: ndarray with shape (num_samples, )
        """
        # Now the input X for clustering is normalized (within the range of [0, 1])
        return self.classifier.predict(X.detach().cpu().numpy())
    
class StandardClassifierIII(StandardClassifier):
    def __init__(self, bounds: torch.Tensor, seed: int, classifier_type: str, classifier_params: Dict, cluster_type: str, cluster_params: Dict, fit_feat_vals: bool = True) -> None:
        super().__init__(bounds, seed, classifier_type, classifier_params, cluster_type, cluster_params, fit_feat_vals)
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> bool:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        
        # labels: ndarray with shape (num_samples, )
        labels, cluster_data = self.cluster(X, Y * 4)
        if len(set(labels)) == 1:
            return False
    
        self.cluster_score = silhouette_score(cluster_data, labels)
        self.classifier = self.classifier.fit(X.detach().cpu().numpy(), labels)
        return True
 

CLASSIFIER_MAP: Dict[str, BaseClassifier] = {
    'base': BaseClassifier,
    'standard': StandardClassifier,
    'standard_ii': StandardClassifierII,
    'standard_iii': StandardClassifierIII,
}