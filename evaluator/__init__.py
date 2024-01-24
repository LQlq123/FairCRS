from loguru import logger

from .conv import ConvEvaluator
from .rec import RecEvaluator
from .standard import StandardEvaluator
from ..data import dataset_language_map

Evaluator_register_table = {
    'rec': RecEvaluator,
    'conv': ConvEvaluator,
    'standard': StandardEvaluator
}


def get_evaluator(evaluator_name, dataset, tensorboard=False):
    if evaluator_name in Evaluator_register_table:
        if evaluator_name in ('conv', 'standard'):
            language = dataset_language_map[dataset]
            evaluator = Evaluator_register_table[evaluator_name](language, tensorboard=tensorboard)
        else:
            evaluator = Evaluator_register_table[evaluator_name](tensorboard=tensorboard)
        logger.info(f'[Build evaluator {evaluator_name}]')
        return evaluator
    else:
        raise NotImplementedError(f'Model [{evaluator_name}] has not been implemented')
