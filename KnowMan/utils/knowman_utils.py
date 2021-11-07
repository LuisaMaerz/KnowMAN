import enum
from transformers import RobertaTokenizer, DistilBertTokenizer
from KnowMan.models.nn_models import RobertaKnowMANFeatureExt, DistilbertKnowMANFeatureExt


class FeatureExtraction(enum.Enum):
    ROBERTABASE = 1
    DISTILBERT = 2

    @staticmethod
    def name2type(text):
        return {'distilbert-base-cased':FeatureExtraction.DISTILBERT,
                'roberta-base': FeatureExtraction.ROBERTABASE}[text]


class TransformerUtil:
    name2type = {'roberta-base': FeatureExtraction.ROBERTABASE,
                 'distilbert-base-cased': FeatureExtraction.DISTILBERT}

    @staticmethod
    def get_tokenizer(transformer_enum):
        if transformer_enum == FeatureExtraction.ROBERTABASE:
            return RobertaTokenizer.from_pretrained('roberta-base')
        if transformer_enum == FeatureExtraction.DISTILBERT:
            return DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    @staticmethod
    def get_pretrained_model(transformer_enum, dropout, out_size):
        if transformer_enum == FeatureExtraction.ROBERTABASE:
            return RobertaKnowMANFeatureExt.from_pretrained('roberta-base', dropout=dropout,
                                                            out_size=out_size)

        if transformer_enum == FeatureExtraction.DISTILBERT:
            return DistilbertKnowMANFeatureExt.from_pretrained('distilbert-base-cased', dropout=dropout,
                                                               out_size=out_size)


def unpackKnowMAN_batch(batch_X, device):
    inputs, labels, adv_labels = batch_X
    return {k: v.to(device) for k, v in inputs.items()}, labels.to(device), adv_labels.to(device)


def freeze_net(net):
    if not net:
        return
    for name, p in net.named_parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for name, p in net.named_parameters():
        if 'transformer' not in name:
            p.requires_grad = True

