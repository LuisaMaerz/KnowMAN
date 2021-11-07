import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from torch import LongTensor
from torch.utils.data import TensorDataset
from knodle.trainer.snorkel.snorkel import SnorkelTrainer
from KnowMan.data_prep.get_knodle_dataset import get_data, get_tfidf_features
from KnowMan.models.nn_models import SentimentClassifier
from KnowMan.utils.knowman_parameters import KnowManParameters


path = os.path.dirname(__file__)
sys.path.append(path + '/../')


def get_dataset(data_path: str, if_dev_data: bool = True):
    # first, the data is read from the file
    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, mapping_rules_labels_t = \
        get_data(data_path, if_dev_data=if_dev_data)

    val = "sample"

    # For the LogReg model we encode train samples with TF-IDF features.
    if if_dev_data:
        train_tfidf, test_tfidf, dev_tfidf = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist(),
                                                                dev_df[val].tolist())

        dev_set = np_array_to_tensor_dataset(dev_tfidf.toarray())
        dev_labels = TensorDataset(LongTensor(dev_df["label"].tolist()))
        dev_dataset = [dev_set, dev_labels]
        train_set = np_array_to_tensor_dataset(train_tfidf.toarray())
        test_set = np_array_to_tensor_dataset(test_tfidf.toarray())
        test_labels = TensorDataset(LongTensor(test_df["label"].tolist()))
        train_dataset = [train_set]
        test_dataset = [test_set, test_labels]

    else:
        train_tfidf, test_tfidf, _ = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist())

        train_set = np_array_to_tensor_dataset(train_tfidf.toarray())
        test_set = np_array_to_tensor_dataset(test_tfidf.toarray())
        test_labels = TensorDataset(LongTensor(test_df["label"].tolist()))
        train_dataset = [train_set]
        test_dataset = [test_set, test_labels]

    if if_dev_data:
        return train_dataset, dev_dataset, test_dataset, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
               mapping_rules_labels_t
    else:
        return train_dataset, test_dataset, train_rule_matches_z, test_rule_matches_z, mapping_rules_labels_t


def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x.float())
    return x


def snorkel_train(params, if_dev_data: bool = True):
    data_path = params.dataset["dataset_path"]
    model = SentimentClassifier(params.model_params["C_layers"],
                                params.model_params["feature_num"],
                                params.model_params["shared_hidden_size"],
                                params.model_params["num_labels"],
                                params.model_params["dropout"],
                                params.model_params["C_bn"])

    if if_dev_data:
        train_dataset, dev_dataset, test_dataset, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
          mapping_rules_labels_t = get_dataset(data_path, if_dev_data)

        trainer = SnorkelTrainer(
            model=model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            model_input_x=train_dataset[0],
            rule_matches_z=train_rule_matches_z,
            dev_model_input_x=dev_dataset[0],
            dev_gold_labels_y=dev_dataset[1]
        )
        trainer.train()
        return trainer.test(test_dataset[0], test_dataset[1])

    else:
        train_dataset, test_dataset, train_rule_matches_z, test_rule_matches_z, mapping_rules_labels_t = \
            get_dataset(data_path, if_dev_data)

        trainer = SnorkelTrainer(
            model=model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            model_input_x=train_dataset[0],
            rule_matches_z=train_rule_matches_z,
        )

        trainer.train()
        return trainer.test(test_dataset[0], test_dataset[1])


def main():
    yaml_file = sys.argv[1]

    params = KnowManParameters()
    params.update_parameters(yaml_file)

    res, gold, pred = snorkel_train(params)
    print(res)

    with open("../KnowMan/save/labels_sorkel_tfidf", "w")as out:
        for i in range(len(gold)):
            out.write(str(gold[i]) + "," + str(pred[i]) + "\n")

    return res


if __name__ == '__main__':
    main()
