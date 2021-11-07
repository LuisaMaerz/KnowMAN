import os
import sys
import time
import torch
import pickle
import logging
import random
import itertools
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from collections import defaultdict
from torch.nn import functional
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter
from sklearn.metrics import classification_report

from KnowMan.utils.knowman_utils import freeze_net, unfreeze_net
from KnowMan.models.nn_models import MlpFeatureExtractor, SentimentClassifier, DomainClassifier
from KnowMan.data_prep.get_knodle_dataset import get_dataset
from KnowMan.utils.logging_utils import per_step_classifier_tb_logging
from KnowMan.utils.knowman_parameters import KnowManParameters


def train(train_set, test_set, params, log):
    """
    train_set, dev_set, test_set: raw_datasets from corpus
    """
    if params.training_setting["use_tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=params.training_setting["tensorboard_dir"],
                               comment='Using lambda {}'.format(params.model_params["lambd"]))
        print(f"tensorboard logging path is {params.training_setting['tensorboard_dir']}")

    train_loader = DataLoader(train_set, params.training_setting["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, params.training_setting["batch_size"], shuffle=True)

    F_s = None
    C, D = None, None

    F_s = MlpFeatureExtractor(params.model_params["feature_num"], params.model_params["F_hidden_sizes"],
                              params.model_params["shared_hidden_size"], params.model_params["dropout"],
                              params.model_params["F_bn"])
    C = SentimentClassifier(params.model_params["C_layers"], params.model_params["shared_hidden_size"] +
                            params.model_params["domain_hidden_size"],
                            params.model_params["shared_hidden_size"] + params.model_params["domain_hidden_size"],
                            params.model_params["num_labels"], params.model_params["dropout"],
                            params.model_params["C_bn"])
    D = DomainClassifier(params.model_params["D_layers"], params.model_params["shared_hidden_size"],
                         params.model_params["shared_hidden_size"],  params.model_params["all_domains"],
                         params.model_params["loss"], params.model_params["dropout"], params.model_params["D_bn"])

    F_s, C, D = F_s.to(params.training_setting["device"]), \
        C.to(params.training_setting["device"]), \
        D.to(params.training_setting["device"])

    optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()])),
                           lr=params.training_setting["learning_rate"])
    optimizerD = optim.Adam(D.parameters(), lr=params.training_setting["D_learning_rate"])

    # testing
    if params.training_setting["test_only"]:
        log.info(f'Loading model from {params.training_setting["model_save_file"]}...')
        F_s.load_state_dict(torch.load(os.path.join(params.training_setting["model_save_file"], f'netF_s.pth')))
        C.load_state_dict(torch.load(os.path.join(params.training_setting["model_save_file"], f'netC.pth')))
        D.load_state_dict(torch.load(os.path.join(params.training_setting["model_save_file"], f'netD.pth')))

        log.info('Evaluating test sets:')
        test_classification_report_dict, y_true, y_pred = evaluate(test_loader, F_s, C, params, log, return_labels=True)
        log.info(f'Average test accuracy: {test_classification_report_dict["accuracy"]}')

        y_true = np.ndarray.tolist(y_true)
        y_pred = np.ndarray.tolist(y_pred)

        with open("../../KnowMan/save/labels_best_spam_tfidf.csv", "w")as out:
            for i in range(len(y_true)):
                out.write(str(y_true[i]) + "," + str(y_pred[i]) + "\n")

        print({'test': test_classification_report_dict["accuracy"]})

    # training
    else:
        best_avg_acc, best_acc = defaultdict(float), 0.0
        batches_between_logging = params.training_setting["batches_between_logging"]
        evaluate_after_batches_between_logging = params.training_setting["evaluate_after_batches_between_logging"]
        num_training_items = len(train_loader)
        print(f"Number of training batches: {num_training_items}")
        total_steps = 0
        avg_train_losses_classifier, avg_train_losses_domain_blurrer, avg_train_losses_classifier_dom_blurrer = 0, 0, 0
        tmp_test_classification_report_dict = evaluate(test_loader, F_s, C, params, log)
        classification_report_dict = evaluate(train_loader, F_s, C, params, log)

        for epoch in range(params.training_setting["max_epoch"]):
            per_epoch_discr_loss_collection = []
            per_epoch_train_losses_classifier = []
            per_epoch_train_losses_domain_blurrer = []
            per_epoch_train_losses_classifier_dom_blurrer = []
            for group in optimizer.param_groups:
                learning_rate = group["lr"]
                writer.add_scalar("learning_rate", learning_rate, epoch)
            for group in optimizerD.param_groups:
                learning_rate = group["lr"]
                writer.add_scalar("learning_rate_D", learning_rate, epoch)

            F_s.train()
            C.train()
            D.train()

            # D iteration
            d_correct, d_total = 0, 0

            freeze_net(F_s)
            freeze_net(C)
            unfreeze_net(D)

            for critic_loop_index in range(params.model_params["n_critic"]):
                discr_loss_collection = []
                for i, (inputs, labels, adv_labels) in enumerate(train_loader):
                    inputs, labels, adv_labels = inputs.to(params.training_setting["device"]), \
                                                 labels.to(params.training_setting["device"]), \
                                                 adv_labels.to(params.training_setting["device"])

                    shared_feat = F_s(inputs)
                    outputs = D(shared_feat)

                    _, pred = torch.max(outputs, 1)
                    d_total += len(inputs)

                    d_correct += (pred == adv_labels).sum().item()
                    l_d = functional.nll_loss(outputs, adv_labels)
                    discr_loss_collection.append(l_d.item())
                    per_epoch_discr_loss_collection.append(l_d.item())
                    l_d.backward()
                    optimizerD.step()
                    D.zero_grad()
                    if (i > 0 and i % batches_between_logging == 0) or i == num_training_items - 1:
                        avg_loss_dom_discr = np.mean(np.array(discr_loss_collection))
                        discr_loss_collection = []
                        log.info("Discriminator training Epoch {}, "
                                 "critic_loop_index {}, Step {}...".format(epoch, critic_loop_index, i))
                        log.info(f'Training loss domain discriminator over the last {batches_between_logging} '
                                 f'batches: {avg_loss_dom_discr}')
                        writer.add_scalar('Training loss domain discriminator accumulated over last batches',
                                          avg_loss_dom_discr, total_steps)
                        if d_total > 0:
                            log.info('Domain Training Accuracy: {}%'.format(100.0 * d_correct / d_total))
                        writer.add_scalar("Domain Training Accuracy", d_correct / d_total, total_steps)

                        writer = per_step_classifier_tb_logging(log, writer, None, tmp_test_classification_report_dict,
                                                                total_steps, batches_between_logging,
                                                                classification_report_dict,
                                                                avg_train_losses_classifier,
                                                                avg_train_losses_domain_blurrer,
                                                                avg_train_losses_classifier_dom_blurrer,
                                                                log_dev_test=True, use_accuracy=True)
                    total_steps += 1

            # F&C iteration
            unfreeze_net(F_s)
            unfreeze_net(C)
            freeze_net(D)
            train_losses_classifier = []
            train_losses_domain_blurrer = []
            train_losses_classifier_dom_blurrer = []
            all_c_pred = []
            all_labels = []
            for i, (inputs, labels, adv_labels) in enumerate(train_loader):
                inputs, labels, adv_labels = inputs.to(params.training_setting["device"]), \
                                             labels.to(params.training_setting["device"]), \
                                             adv_labels.to(params.training_setting["device"])
                shared_feat = F_s(inputs)
                c_outputs = C(shared_feat)
                d_outputs = D(shared_feat)

                _, c_pred = torch.max(c_outputs, 1)
                all_c_pred.append(c_pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                l_c = functional.nll_loss(c_outputs, labels)
                train_losses_classifier.append(l_c.item())
                train_losses_domain_blurrer.append(-l_d.item())
                per_epoch_train_losses_classifier.append(l_c.item())
                per_epoch_train_losses_domain_blurrer.append(-l_d.item())

                l_d = functional.nll_loss(d_outputs, adv_labels)
                l_d *= params.model_params["lambd"]

                l_shared = l_c - l_d
                train_losses_classifier_dom_blurrer.append(l_shared.item())
                per_epoch_train_losses_classifier_dom_blurrer.append(l_shared.item())
                l_shared.backward()

                optimizer.step()
                F_s.zero_grad()
                C.zero_grad()

                if (i > 0 and i % batches_between_logging == 0) or i == num_training_items-1:
                    avg_train_losses_classifier_dom_blurrer = np.mean(np.array(train_losses_classifier_dom_blurrer))
                    train_losses_classifier_dom_blurrer = []
                    avg_train_losses_classifier = np.mean(np.array(train_losses_classifier))
                    train_losses_classifier = []
                    avg_train_losses_domain_blurrer = np.mean(np.array(train_losses_domain_blurrer))
                    train_losses_domain_blurrer = []
                    log.info("Classifier training Epoch {}, Step: {}...".format(epoch, i))
                    all_c_pred_array = np.concatenate(all_c_pred, axis=0)
                    all_labels_array = np.concatenate(all_labels, axis=0)
                    classification_report_dict = classification_report(y_true=all_labels_array, y_pred=all_c_pred_array,
                                                                       output_dict=True)
                    t = time.localtime()
                    log.info("Time: {}".format(time.strftime("%H:%M:%S", t)))
                    if d_total > 0:
                        log.info('Domain Training Accuracy: {}%'.format(100.0 * d_correct / d_total))
                    writer.add_scalar('Training loss domain discriminator accumulated over last batches',
                                      avg_loss_dom_discr, total_steps)
                    writer.add_scalar("Domain Training Accuracy", d_correct / d_total, total_steps)

                    if evaluate_after_batches_between_logging or i == num_training_items - 1:
                        tmp_test_classification_report_dict = evaluate(test_loader, F_s, C, params, log)

                    writer = per_step_classifier_tb_logging(log, writer, None, tmp_test_classification_report_dict,
                                                            total_steps, batches_between_logging,
                                                            classification_report_dict,
                                                            avg_train_losses_classifier,
                                                            avg_train_losses_domain_blurrer,
                                                            avg_train_losses_classifier_dom_blurrer,
                                                            log_dev_test=(evaluate_after_batches_between_logging
                                                                          or i == num_training_items - 1),
                                                            use_accuracy=True)

                    if evaluate_after_batches_between_logging or i == num_training_items - 1:
                        if classification_report_dict['accuracy'] > best_acc:
                            best_avg_acc['test'] = tmp_test_classification_report_dict['accuracy']
                            best_acc = classification_report_dict['accuracy']
                            with open(os.path.join(params.training_setting["model_save_file"], 'options.pkl'), 'wb')\
                                    as ouf:
                                pickle.dump(params.get_config(), ouf)
                            log.info("Saving new model")
                            torch.save(F_s.state_dict(), '{}/netF_s.pth'.format(params.training_setting
                                                                                ["model_save_file"]))
                            torch.save(C.state_dict(), '{}/netC.pth'.format(params.training_setting["model_save_file"]))
                            torch.save(D.state_dict(), '{}/netD.pth'.format(params.training_setting["model_save_file"]))

                total_steps += 1

            # end of epoch
            log.info('Ending epoch {}'.format(epoch+1))

        # end of training
        log.info(f'Best average accuracy: {100.0*best_acc}%')
        return best_avg_acc


def evaluate(loader, F_s, C, params, log, return_labels=None):
    F_s.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    all_c_pred = []
    all_labels = []
    confusion = ConfusionMeter(params.model_params["num_labels"])
    for elem in tqdm(it):
        inputs, targets = elem[0], elem[1]
        inputs, targets = inputs.to(params.training_setting["device"]), targets.to(params.training_setting["device"])
        features = F_s(inputs)
        outputs = C(features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
        all_c_pred.append(pred.cpu().numpy())
        all_labels.append(targets.cpu().numpy())
    all_c_pred_array = np.concatenate(all_c_pred, axis=0)
    all_labels_array = np.concatenate(all_labels, axis=0)
    classification_report_dict = classification_report(y_true=all_labels_array, y_pred=all_c_pred_array,
                                                       output_dict=True)
    log.debug(confusion.conf)
    F_s.train()
    C.train()
    if return_labels:
        return classification_report_dict, all_labels_array, all_c_pred_array
    else:
        return classification_report_dict


def set_seeds(params):
    random.seed(params.training_setting["random_seed"])
    np.random.seed(params.training_setting["random_seed"])
    torch.manual_seed(params.training_setting["random_seed"])
    torch.cuda.manual_seed_all(params.training_setting["random_seed"])


def set_logging(params):
    # save models and logging
    if not os.path.exists(params.training_setting["model_save_file"]):
        os.makedirs(params.training_setting["model_save_file"])
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if params.training_setting["debug"] else logging.INFO)
    log = logging.getLogger(__name__)
    fh = logging.FileHandler(os.path.join(params.training_setting["model_save_file"], 'log.txt'))
    log.addHandler(fh)

    # output options
    log.info(params.get_config())

    return log


def main():
    spam_params = KnowManParameters()
    spam_params.update_parameters("./spam_tfidf.yaml")

    set_seeds(spam_params)
    log = set_logging(spam_params)

    if not os.path.exists(spam_params.training_setting["model_save_file"]):
        os.makedirs(spam_params.training_setting["model_save_file"])

    train_dataset, test_dataset = get_dataset(spam_params.dataset["dataset_path"], if_dev_data=False)

    cv = train(train_dataset, test_dataset, spam_params, log)
    log.info(f'Training done...')
    test_acc = cv['test']
    log.info(f'Test Set \t{100.0*test_acc}%')
    return cv


if __name__ == '__main__':
    main()
