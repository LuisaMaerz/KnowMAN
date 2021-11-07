import os
import sys
import numpy as np
from KnowMan.data_prep.get_knodle_dataset import get_data, get_tfidf_features, \
    z_t_matrices_to_majority_vote_probs, probabilities_to_majority_vote
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from KnowMan.utils.knowman_parameters import KnowManParameters


path = os.path.dirname(__file__)


def get_dataset(data_path: str, if_dev_data: bool = True):

    # first, the data is read from the file
    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, \
        mapping_rules_labels_t = get_data(data_path, if_dev_data=if_dev_data)
    if "tac" in data_path:
        val = "samples"
    else:
        val = "sample"

    # For the LogReg model we encode train samples with TF-IDF features.
    if if_dev_data:
        train_tfidf, test_tfidf, dev_tfidf = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist(),
                                                                dev_df[val].tolist())

    else:
        train_tfidf, test_tfidf, _ = get_tfidf_features(train_df[val].tolist(), test_df[val].tolist())

    test_labels = np.asarray(list(test_df.iloc[:, 1]))
    train_probs = z_t_matrices_to_majority_vote_probs(train_rule_matches_z, mapping_rules_labels_t)
    train_labels = np.asarray(np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=train_probs,
                                                  choose_random_label=True))

    if if_dev_data:
        dev_labels = np.asarray(list(dev_df.iloc[:, 1]))

        return [train_tfidf, train_labels], [dev_tfidf, dev_labels], [test_tfidf, test_labels]
    else:
        return [train_tfidf, train_labels], [test_tfidf, test_labels]


def main():
    yaml_file = sys.argv[1]

    params = KnowManParameters()
    params.update_parameters(yaml_file)

    dataset_path = params.dataset["dataset_path"]
    train_dataset, test_dataset = get_dataset(dataset_path, if_dev_data=False)

    clf = LogisticRegression(random_state=0, max_iter=200).fit(train_dataset[0], train_dataset[1])
    pred = clf.predict(test_dataset[0])

    with open("../KnowMan/save/labels_KS_tfidf.csv", "w") as out:
        for i in range(len(test_dataset[1])):
            out.write(str(test_dataset[1][i]) + "," + str(pred[i]) + "\n")

    print(classification_report(test_dataset[1], pred))


if __name__ == '__main__':
    main()
