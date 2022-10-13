"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
XXXXX
***
SingletonClass.py
@author: @XXXXX
"""

import abc

class SingletonClass(metaclass=abc.ABCMeta):
    """
    Singleton class.
    """

    _INSTANCE = None
    
    
    # static methods

    @abc.abstractmethod
    def get_instance():
        pass

    def get_instance(cls):
        """
        Returns the single instance of this class.
        
        Keyword arguments: 
        cls -- class inheriting from SingletonClass from which an instanced is requested.
        
        Return: 
        Single instance of this class.
        """
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(caller='get_instance')
        return cls._INSTANCE