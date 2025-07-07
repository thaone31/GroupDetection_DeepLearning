from abc import ABC, abstractmethod
import networkx as nx
import numpy as np

class CommunityDetectionModel:
    def fit(self, G):
        raise NotImplementedError

    def get_embedding(self):
        raise NotImplementedError

    def predict(self, n_clusters):
        raise NotImplementedError