from abc import ABC, abstractmethod

class DBConnector(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def fetch_results(self, sql, json=True):
        pass

    @abstractmethod
    def execute(self, sql):
        pass
