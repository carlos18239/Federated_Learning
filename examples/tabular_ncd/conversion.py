from typing import Dict, List, Callable
import numpy as np
import torch
import inspect
from .cnn import MLP


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
        self.order_list: List[str] = []
        self.model_ctor: Callable[..., torch.nn.Module] = None

    def set_model_ctor(self, ctor: Callable[..., torch.nn.Module]):
        """Set the model constructor for creating new instances"""
        self.model_ctor = ctor

    def _infer_in_features_from_models(self, models: Dict[str, np.ndarray]) -> int:
        """Infer input features from the first weight matrix"""
        # Try to get first layer weights
        for key in ['fc1_0', 'fc_0', 'layer1_0']:
            arr = models.get(key, None)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return int(arr.shape[1])
        
        # Fallback: find any 2D array
        for k, a in models.items():
            if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] > 0:
                return int(a.shape[1])
        
        raise ValueError("Could not infer in_features from model weights")

    def _ordered_keys(self, models: Dict[str, np.ndarray]) -> List[str]:
        """Return ordered keys respecting the order_list if available"""
        if self.order_list:
            return list(self.order_list)
        
        # Sort by layer name and parameter index
        def ksort(k: str):
            parts = k.split("_")
            base = parts[0]
            try:
                idx = int(parts[1])
            except Exception:
                idx = 0
            return (base, idx)
        
        return sorted(models.keys(), key=ksort)

    def convert_nn_to_dict_nparray(self, net) -> Dict[str, np.ndarray]:
        """
        Convert neural network to a dictionary of np.array models
        and save the order of models in a list (using the names)
        """
        d = dict()
        self.order_list = []  # Reset for this conversion

        # get a dictionary of all layers in the NN
        layers = vars(net)['_modules']

        # for each layer
        for lname, model in layers.items():
            # convert it to numpy
            for i, ws in enumerate(model.parameters()):
                mname = f'{lname}_{i}'
                d[mname] = ws.data.detach().cpu().numpy()
                self.order_list.append(mname)
        
        return d

    def convert_dict_nparray_to_nn(self, models: Dict[str, np.ndarray]):
        """
        Convert np array models in a dict to neural network
        """
        assert self.model_ctor is not None, "Model constructor not set in Converter"

        # Infer input features from weights
        inferred_in = self._infer_in_features_from_models(models)

        # Instantiate model respecting constructor signature
        try:
            sig = inspect.signature(self.model_ctor)
            if len(sig.parameters) >= 1:
                net = self.model_ctor(inferred_in)
            else:
                net = self.model_ctor()
        except TypeError:
            net = self.model_ctor()

        layers = vars(net)['_modules']
        keys = self._ordered_keys(models)
        npa_iter = iter(models[k] for k in keys)

        # for each layer
        for lname, model in layers.items():
            # for loop to separately update w and b
            for ws in model.parameters():
                arr = next(npa_iter)
                t = torch.from_numpy(arr).to(ws.data.dtype)
                ws.data = t.clone().detach()

        return net

    def get_model_names(self, net) -> List[str]:
        """
        Return a list of suggested model names for the config file
        """
        print('=== Model Names (for the config file) ===')
        d = self.convert_nn_to_dict_nparray(net)
        print(d.keys())

        return list(d.keys())