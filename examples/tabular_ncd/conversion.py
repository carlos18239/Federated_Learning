
from typing import Dict, List, Callable
import numpy as np
import torch
from .cnn import MLP
from typing import Dict, List



class Converter:
    """
    Conversion functions
    Note: Singleton Pattern
    """
    _singleton_cvtr = None

    @classmethod
    def cvtr(cls):
        if not cls._singleton_cvtr:
            cls._singleton_cvtr = cls()
        return cls._singleton_cvtr

    def __init__(self):
        # This list saves the order of models (using the predefined model names)
        self.order_list = list()

    def convert_nn_to_dict_nparray(self, MLP) -> Dict[str, np.array]:
        """
        Convert CNN to a dictionary of np.array models
        and save the order of models in a list (using the names)
        :param MLP:
        :return:
        """
        d = dict()

        # get a dictionary of all layers in the NN
        layers = vars(MLP)['_modules']

        # for each layer
        for lname, model in layers.items():
            # convert it to numpy
            if not lname == 'pool':
                for i, ws in enumerate(model.parameters()):
                    mname = f'{lname}_{i}'
                    d[mname] = ws.data.numpy()
                    self.order_list.append(mname)
        return d

    def convert_dict_nparray_to_nn(self, models: Dict[str, np.array]) -> MLP:
        """
        Convert np array models in a dict to CNN
        :param models:
        :return: CNN (MLP class)
        """
        MLP = MLP()
        layers = vars(MLP)['_modules']

        npa_iter = iter(_order_dict(models, self.order_list))

        # for each layer
        for lname, model in layers.items():
            if not lname == 'pool':
                # for loop to separatly update w and b
                for ws in model.parameters():
                    # Since the order is kept in NN
                    # update it
                    ws.data = torch.from_numpy(next(npa_iter))

        return MLP

    def get_model_names(self, MLP) -> List[str]:
        """
        Return a list of suggested model names for the config file
        :param MLP:
        :return:
        """
        print('=== Model Names (for the config file) ===')
        d = self.convert_nn_to_dict_nparray(MLP)
        print(d.keys())

        return d.keys()


def _order_dict(d: Dict, l: List) -> List:
    """
    Return a list of np.array ordered based on the list l
    :param d: Models (np.arrays) in a dict
    :param l: The order of models (each element is a key of the dict d
    :return: List of models (np.arrays)
    """
    ordered_vals = list()
    for key in l:
        ordered_vals.append(d[key])
    return ordered_vals
