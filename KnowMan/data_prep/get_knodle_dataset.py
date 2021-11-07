import os
import sys
import numpy as np
from joblib import load
import scipy.sparse as sp
from typing import Union, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset
from KnowMan.utils.vocab import KnowMANDataSet
from KnowMan.utils.knowman_utils import FeatureExtraction, TransformerUtil

path = os.path.dirname(__file__)
sys.path.append(path+'/../')


def get_data(target_path: str, if_dev_data: bool = True):
    """
    Reads data, label and also retrieves adversarial label
    Returns TensorDataset to be used in DataLoader
    train_data, dev_data, test_data: csv. file with index, sample, label
    t_train, t_dev: dataframe with mapping rules - labels
    z_train, z_dev: dataframe with mapping instances - rules
    """
    train_df = load(os.path.join(target_path, 'df_train.lib'))
    test_df = load(os.path.join(target_path, 'df_test.lib'))

    if "imdb" in target_path:
        train_rule_matches_z = load(os.path.join(target_path, 'train_rule_matches_z.lib')).toarray()
        test_rule_matches_z = load(os.path.join(target_path, 'test_rule_matches_z.lib')).toarray()
    else:
        train_rule_matches_z = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
        test_rule_matches_z = load(os.path.join(target_path, 'test_rule_matches_z.lib'))

    mapping_rules_labels_t = load(os.path.join(target_path, 'mapping_rules_labels_t.lib'))

    if if_dev_data:
        dev_df = load(os.path.join(target_path, 'df_dev.lib'))
        if "imdb" in target_path:
            dev_rule_matches_z = load(os.path.join(target_path, 'dev_rule_matches_z.lib')).toarray()
        else:
            dev_rule_matches_z = load(os.path.join(target_path, 'dev_rule_matches_z.lib'))
        return train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
            mapping_rules_labels_t

    return train_df, None, test_df, train_rule_matches_z, None, test_rule_matches_z, mapping_rules_labels_t


def get_tfidf_features(
        train_data: List, test_data: List = None, dev_data: List = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """
    Convert input data to a matrix of TF-IDF features.
    :param train_data: training samples that are to be encoded with TF-IDF features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples
    are stored
    :param test_data: if DataFrame/Series with test data is provided
    :param dev_data: if DataFrame/Series with development data is provided, it will be encoded as well
    :return: TensorDataset with encoded data
    """
    dev_transformed_data, test_transformed_data = None, None
    vectorizer = TfidfVectorizer()

    train_transformed_data = vectorizer.fit_transform(train_data)
    if test_data is not None:
        test_transformed_data = vectorizer.transform(test_data)
    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(dev_data)
    return train_transformed_data, test_transformed_data, dev_transformed_data


def z_t_matrices_to_majority_vote_probs(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, other_class: int = None
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z. The difference from simple
    get_majority_vote_probs function is the following: samples, where no rules matched (that is, all elements in
    the corresponding row in rule_matches_z matrix equal 0), are assigned to no_match_class (that is, a value in the
    corresponding column in rule_counts_probs matrix is changed to 1).
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        other_class: Class which is chosen, if no function is hitting.
    Returns: Array with majority vote probabilities. Shape: instances x classes
    """

    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError(f"Dimensions mismatch! Z matrix has shape {rule_matches_z.shape}, while "
                         f"T matrix has shape {mapping_rules_labels_t.shape}")

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_counts = rule_matches_z.dot(mapping_rules_labels_t).toarray()
    else:
        rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)

    if other_class:
        if other_class < 0:
            raise RuntimeError("Label for negative samples should be greater than 0 for correct matrix multiplication")
        if other_class < mapping_rules_labels_t.shape[1] - 1:
            warnings.warn(f"Negative class {other_class} is already present in data")
        if rule_counts.shape[1] == other_class:
            rule_counts = np.hstack((rule_counts, np.zeros([rule_counts.shape[0], 1])))
            rule_counts[~rule_counts.any(axis=1), other_class] = 1
        elif rule_counts.shape[1] >= other_class:
            rule_counts[~rule_counts.any(axis=1), other_class] = 1
        else:
            raise ValueError("Other class id is incorrect")
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)
    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def probabilities_to_majority_vote(
        probs: np.ndarray, choose_random_label: bool = False, other_class_id: int = None,
        multiple_instances: bool = False
) -> int:
    """Transforms a vector of probabilities to its majority vote. If there is one class with clear majority, return it.
    If there are more than one class with equal probabilities: either select one of the classes randomly, return a
    vector containing all of them or assign to the sample the other class id.

    Args:
        probs: Vector of probabilities for 1 sample. Shape: classes x 1
        choose_random_label: Choose a random label, if there's no clear majority.
        other_class_id: Class ID being used, if there's no clear majority
        multiple_instances: Return duplicated instances with labels, if there are several maxima.
    Returns: An array of classes.
    """
    if choose_random_label and other_class_id is not None:
        raise ValueError("You can either choose a random class, or transform undefined cases to an other class.")
    if choose_random_label and multiple_instances:
        raise ValueError("You can either choose a random class, or create multiple instances with multiple classes.")

    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:
        return int(np.argmax(probs))
    elif choose_random_label:
        max_ids = np.where(probs == row_max)[0]
        return int(np.random.choice(max_ids))
    elif multiple_instances:
        return np.where(probs == row_max)[0]
    elif other_class_id is not None:
        return other_class_id
    else:
        raise ValueError("Specify a way how to resolve unclear majority votes.")


def probabilies_to_majority_class_label(
        probs: np.ndarray, choose_random_label: bool = False, other_class_id: int = None,
        multiple_instances: bool = False,
) -> int:
    """Transforms a vector of probabilities to its majority vote. If there is one class with clear majority, return it.
    If there are more than one class with equal probabilities: either select one of the classes randomly or assign to
    the sample the other class id.

    Args:
        probs: Vector of probabilities for 1 sample. Shape: classes x 1
        choose_random_label: Choose a random label, if there's no clear majority.
        other_class_id: Class ID being used, if there's no clear majority
        multiple_instances: Return duplicated instances with labels, if there are more maxima.
    Returns: An array of classes.
    """
    if choose_random_label and other_class_id is not None:
        raise ValueError("You can either choose a random class, or transform undefined cases to an other class.")
    if choose_random_label and multiple_instances:
        raise ValueError("You can either choose a random class, or create multiple instances with multiple classes.")

    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:
        return int(np.argmax(probs))
    elif choose_random_label:
        max_ids = np.where(probs == row_max)[0]
        return int(np.random.choice(max_ids))
    elif multiple_instances:
        return np.where(probs == row_max)[0]
    elif other_class_id is not None:
        return other_class_id
    else:
        raise ValueError("Specify a way how to resolve unclear majority votes.")


def z_matrix_to_rule_idx(
        rules: np.ndarray, choose_random_rule: bool = False, multiple_instances: bool = False
) -> int:
    """Transforms a z matrix to rule indices of matching rules.
    If there is more than one rule match: either select one of the rules randomly or return a vector containing
    all of them.

    Args:
        rules: Vector of probabilities for 1 sample. Shape: classes x 1
        choose_random_rule: Choose a random label, if there's no clear majority.
        multiple_instances: Return duplicated instances with idx, if there are several rule matches.
    Returns: An array of classes.
    """
    if choose_random_rule and multiple_instances:
        raise ValueError("You can either choose a random rule, or create multiple instances with multiple rules.")

    row_max = np.max(rules)
    num_occurrences = (row_max == rules).sum()
    if num_occurrences == 1:
        return int(np.argmax(rules))
    elif choose_random_rule:
        max_ids = np.where(rules == row_max)[0]
        return int(np.random.choice(max_ids))
    elif multiple_instances:
        return np.where(rules == row_max)[0]
    else:
        raise ValueError("Specify a way how to resolve multiple rule matches.")


def get_dataset(data_path: str, use_tfidf: bool = True, if_dev_data: bool = True):
    # first, the data is read from the file
    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
        mapping_rules_labels_t = get_data(data_path, if_dev_data=if_dev_data)
    if "tac" in data_path:
        val = "samples"
    else:
        val = "sample"

    if use_tfidf:
        # For the LogReg model we encode train samples with TF-IDF features.
        if if_dev_data:
            train_tfidf, test_tfidf, dev_tfidf = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist(),
                                                                    dev_df[val].tolist())

        else:
            train_tfidf, test_tfidf, _ = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist())

        train_set = Tensor(train_tfidf.toarray())
        test_set = Tensor(test_tfidf.toarray())
    else:
        train_set = train_df[val].tolist()
        test_set = test_df[val].tolist()

    test_labels = LongTensor(list(test_df.iloc[:, 1]))
    train_probs = z_t_matrices_to_majority_vote_probs(train_rule_matches_z, mapping_rules_labels_t)
    train_labels = LongTensor(np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=train_probs,
                                                  choose_random_label=True))
    train_adv_labels = LongTensor(np.apply_along_axis(z_matrix_to_rule_idx, axis=1,
                                                      arr=train_rule_matches_z, choose_random_rule=True))

    train_dataset = TensorDataset(train_set, train_labels, train_adv_labels)
    test_dataset = TensorDataset(test_set, test_labels)

    if if_dev_data:
        if use_tfidf:
            dev_set = Tensor(dev_tfidf.toarray())
        else:
            dev_set = dev_df[val].tolist()
        dev_labels = LongTensor(list(dev_df.iloc[:, 1]))
        dev_adv_labels = LongTensor(np.apply_along_axis(z_matrix_to_rule_idx, axis=1,
                                                        arr=dev_rule_matches_z, choose_random_rule=True))
        dev_dataset = TensorDataset(dev_set, dev_labels, dev_adv_labels)
        return train_dataset, dev_dataset, test_dataset
    else:
        return train_dataset, test_dataset


def get_transformer_dataset(data_path: str, feature_ext: FeatureExtraction = FeatureExtraction.DISTILBERT,
                            if_dev_data: bool = True, max_length_transformer=200):

    # first, the data is read from the file
    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
      mapping_rules_labels_t = get_data(data_path, if_dev_data=if_dev_data)

    if "tac" in data_path:
        val = "samples"
    else:
        val = "sample"

    tokenizer = TransformerUtil.get_tokenizer(feature_ext)
    train_set = tokenizer(train_df[val].tolist(), truncation=True, padding='max_length')
    test_set = tokenizer(test_df[val].tolist(), truncation=True, padding='max_length')

    test_labels = LongTensor(list(test_df.iloc[:, 1]))
    train_probs = z_t_matrices_to_majority_vote_probs(train_rule_matches_z, mapping_rules_labels_t)
    train_labels = LongTensor(np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=train_probs,
                                                  choose_random_label=True))
    train_adv_labels = LongTensor(np.apply_along_axis(z_matrix_to_rule_idx, axis=1,
                                                      arr=train_rule_matches_z, choose_random_rule=True))

    train_dataset = KnowMANDataSet(train_set, train_labels, train_adv_labels, max_length=max_length_transformer)
    test_dataset = KnowMANDataSet(test_set, test_labels, max_length=max_length_transformer)

    if if_dev_data:
        dev_set = tokenizer(dev_df[val].tolist(), truncation=True, padding='max_length')

        dev_labels = LongTensor(list(dev_df.iloc[:, 1]))
        dev_adv_labels = LongTensor(np.apply_along_axis(z_matrix_to_rule_idx, axis=1,
                                                        arr=dev_rule_matches_z, choose_random_rule=True))

        dev_dataset = KnowMANDataSet(dev_set, dev_labels, dev_adv_labels, max_length=max_length_transformer)

        return train_dataset, dev_dataset, test_dataset
    else:
        return train_dataset, test_dataset
