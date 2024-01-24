from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base class for evaluator"""

    def rec_evaluate(self, preds, label):
        pass

    def gen_evaluate(self, preds, label):
        pass

    def policy_evaluate(self, preds, label):
        pass

    @abstractmethod
    def report(self, epoch, mode):
        pass

    @abstractmethod
    def reset_metrics(self):
        pass
