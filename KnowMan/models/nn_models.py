from KnowMan.models.layers import *
import torch.nn.functional as functional
from transformers import RobertaForSequenceClassification, DistilBertForSequenceClassification


class MlpFeatureExtractor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(MlpFeatureExtractor, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential()
        num_layers = len(hidden_sizes)
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                self.net.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.net.add_module('f-relu-{}'.format(i), nn.ReLU())

        if dropout > 0:
            self.net.add_module('f-dropout-final', nn.Dropout(p=dropout))
        self.net.add_module('f-linear-final', nn.Linear(hidden_sizes[-1], output_size))
        if batch_norm:
            self.net.add_module('f-bn-final', nn.BatchNorm1d(output_size))
        self.net.add_module('f-relu-final', nn.ReLU())

    def forward(self, input):
        return self.net(input)


class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class DomainClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_domains,
                 loss_type,
                 dropout,
                 batch_norm=False):
        super(DomainClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_domains = num_domains
        self.loss_type = loss_type
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        if loss_type.lower() == 'gr' or loss_type.lower() == 'bs':
            self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        scores = self.net(input)
        if self.loss_type.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, linear=False):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_layer = nn.Linear(self.input_size, self.hidden_size)
        self.is_linear = linear
        if not linear:
            self.relu = nn.ReLU()

    def forward(self, x):
        projection = self.linear_layer(x)
        if self.is_linear:
            return projection

        features = self.relu(projection)
        return features


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes, linear=False):
        super(Classifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.linear_layer = nn.Linear(feature_dim, num_classes)
        self.is_linear = linear
        if not self.is_linear:
            self.relu = nn.ReLU()

    def forward(self, x):
        projection = self.linear_layer(x)
        if self.is_linear:
            return projection
        classification = self.relu(projection)
        return classification


class RobertaKnowMANFeatureExt(RobertaForSequenceClassification):
    """
        Roberta feature extractor
    """

    def __init__(self, config, dropout=0.5, out_size=100):
        super().__init__(config)
        self.KnowMANFeat_layer = TransformerHeadFeature(config, dropout, out_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        return self.KnowMANFeat_layer(sequence_output)


class DistilbertKnowMANFeatureExt(DistilBertForSequenceClassification):
    """
        Roberta feature extractor
    """

    def __init__(self, config, drop_out=0.5, out_size=100):
        super().__init__(config)
        self.dropout = nn.Dropout(drop_out)
        self.pre_classifier = nn.Linear(config.dim, out_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        return pooled_output
