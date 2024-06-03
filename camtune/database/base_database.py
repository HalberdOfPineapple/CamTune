from abc import ABC, abstractmethod
from ConfigSpace.configuration_space import Configuration 

class BaseDatabase(ABC):
    remote_mode: bool

    @abstractmethod
    def __init__(self, args_db: dict, llama_params: dict = None, on_wsl: bool = False):
        raise NotImplementedError

    @abstractmethod
    def clear_sys_states(self):
        raise NotImplementedError

    @abstractmethod
    def clear_db_states(self, clear_db_data: bool = False):
        raise NotImplementedError
    
    @abstractmethod
    def default_restart(self, exec: bool =False, dummy: bool = False):
        raise NotImplementedError
    
    @abstractmethod
    def apply_knob(self, config: Configuration, dummy: bool = False):
        raise NotImplementedError

    @abstractmethod
    def step(self, config: Configuration, exec_overwrite: str = None, dummy: bool = False) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reboot(self):
        raise NotImplementedError