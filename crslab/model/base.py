from abc import ABC, abstractmethod

from torch import nn

from crslab.download import build


class BaseModel(ABC, nn.Module):
    """Base class for all models
    （参数初始化和模型加载）的基本属性和函数统一到类BaseModel"""

    def __init__(self, opt, device, dpath=None, resource=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device

        if resource is not None:
            self.dpath = dpath
            dfile = resource['file']
            build(dpath, dfile, version=resource['version'])

        self.build_model()

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """build model"""
        pass

    def recommend(self, batch, mode):
        """calculate loss and prediction of recommendation for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass

    def converse(self, batch, mode):
        """calculate loss and prediction of conversation for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass

    def guide(self, batch, mode):
        """calculate loss and prediction of guidance for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass
