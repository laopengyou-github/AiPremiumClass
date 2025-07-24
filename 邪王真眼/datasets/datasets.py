from enum import Enum
from typing import Any, Dict
from torch.utils.data import DataLoader


class DatasetType(Enum):
    DOUBAN_COMMENTS = 'douban_comments'
    CIFAR10 = "cifar10"
    SEM_EVAL = "sem_eval"
    SKIP_GRAM = "skip_gram"
    CBOW = "cbow"
    Corpus_P6 = "corpus_p6"
    E_Commerce_Comments = "e_commerce_comments"
    E_Commerce_Comments_Idx = "e_commerce_comments_idx"
    BREAST_CANCER = "breast_cancer"
    KMNIST = "kmnist"
    OLIVETTI_FACES = "olivetti_faces"
    NER = "ner"
    Weather = "weather"
    COUPLET = "couplet"
    NextWordPrediction = "NextWordPrediction"
    NextWordPrediction2 = "NextWordPrediction2"
    QA = "qa"
    QA_ENCODER_DECODER = "qa_encoder_decoder"
    JD_COMMENTS = "jd_comments"

    @classmethod
    def from_str(cls, label: str) -> "DatasetType":
        if label in cls.__members__:
            return cls[label]
        
        for member in cls:
            if member.value.lower() == label.lower():
                return member
        raise ValueError(f"Unknown DatasetType: {label!r}. "
                         f"Valid names: {list(cls.__members__.keys())}, "
                         f"values: {[m.value for m in cls]}")


def get_dataset(cfg: Dict[str, Any]):
    dataset_type = cfg.get('dataset')
    if isinstance(dataset_type, str):
        dataset_type = DatasetType.from_str(dataset_type)
    batch_size = cfg.get('batch_size')
    collate_fn = None

    if dataset_type == DatasetType.DOUBAN_COMMENTS:
        from .douban_comments.generator import DoubanCommentDataset
        train_dataset = DoubanCommentDataset('train', cfg)
        val_dataset = DoubanCommentDataset('val', cfg)
    elif dataset_type == DatasetType.CIFAR10:
        from .cifar10.generator import CIFAR10Dataset
        train_dataset = CIFAR10Dataset('train', cfg)
        val_dataset = CIFAR10Dataset('val', cfg)
    elif dataset_type == DatasetType.SEM_EVAL:
        from .sem_eval.generator import SemEvalDataset
        train_dataset = SemEvalDataset('train', cfg)
        val_dataset = SemEvalDataset('val', cfg)
    elif dataset_type == DatasetType.SKIP_GRAM:
        from .word2vec.generator_skip_gram import Skipgram
        train_dataset = Skipgram('train', cfg)
        val_dataset = Skipgram('val', cfg)
    elif dataset_type == DatasetType.CBOW:
        from .word2vec.generator_cbow import CBOW
        train_dataset = CBOW('train', cfg)
        val_dataset = CBOW('val', cfg)
    elif dataset_type == DatasetType.Corpus_P6:
        from .corpus_p6.generator import CorpusP6
        train_dataset = CorpusP6('train', cfg)
        val_dataset = CorpusP6('val', cfg)
    elif dataset_type == DatasetType.E_Commerce_Comments:
        from .e_commerce_comments.generator import ECommerceComments
        train_dataset = ECommerceComments('train', cfg)
        val_dataset = ECommerceComments('val', cfg)
    elif dataset_type == DatasetType.E_Commerce_Comments_Idx:
        from .e_commerce_comments.generator_idx import ECommerceComments
        train_dataset = ECommerceComments('train', cfg)
        val_dataset = ECommerceComments('val', cfg)
    elif dataset_type == DatasetType.BREAST_CANCER:
        from .breast_cancer.generator import BreastCancer
        train_dataset = BreastCancer('train', cfg)
        val_dataset = BreastCancer('val', cfg)
    elif dataset_type == DatasetType.KMNIST:
        from .kmnist.generator import KMNIST
        train_dataset = KMNIST('train', cfg)
        val_dataset = KMNIST('val', cfg)
    elif dataset_type == DatasetType.OLIVETTI_FACES:
        from .olivetti_faces.generator import OlivettiFaces
        train_dataset = OlivettiFaces('train', cfg)
        val_dataset = OlivettiFaces('val', cfg)
    elif dataset_type == DatasetType.NER:
        from .corpus_p9.generator import NER
        train_dataset = NER('train', cfg)
        val_dataset = NER('val', cfg)
    elif dataset_type == DatasetType.Weather:
        from .weather.generator import Weather
        dataset = Weather(cfg)
        train_dataset, val_dataset = dataset.get_datasets()
    elif dataset_type == DatasetType.COUPLET:
        from .couplet.generator import Couplet
        train_dataset = Couplet('train', cfg)
        val_dataset = Couplet('val', cfg)
        collate_fn = Couplet.collate_fn
    elif dataset_type == DatasetType.NextWordPrediction:
        from .corpus_p10.generator import NextWordPrediction
        train_dataset = NextWordPrediction('train', cfg)
        val_dataset = NextWordPrediction('val', cfg)
    elif dataset_type == DatasetType.NextWordPrediction2:
        from .corpus_16.generator import NextWordPrediction
        train_dataset = NextWordPrediction(cfg)
        val_dataset = NextWordPrediction(cfg)
    elif dataset_type == DatasetType.QA:
        from .corpus_p11.generator import QA
        train_dataset = QA('train', cfg)
        val_dataset = QA('val', cfg)
    elif dataset_type == DatasetType.QA_ENCODER_DECODER:
        from .corpus_p11.generator_en_de import QA
        train_dataset = QA('train', cfg)
        val_dataset = QA('val', cfg)
    elif dataset_type == DatasetType.JD_COMMENTS:
        from .jd_comments.generator import JDComments
        train_dataset = JDComments('train', cfg)
        val_dataset = JDComments('val', cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
