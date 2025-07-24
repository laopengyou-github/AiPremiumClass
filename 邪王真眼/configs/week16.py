from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.Nano_GPT,
    n_layer = 12,
    n_head = 8,
    n_embd = 768,
    dropout = 0.1,

    dataset = DatasetType.NextWordPrediction2,
    batch_size = 64,
    data_root = './邪王真眼/datasets/corpus_16',
    max_length = 20,
    num_samples = 1280,

    optimizer = OptimizerType.AdamW,
    lr = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 50,
    result_path = './邪王真眼/week16/result'
)
