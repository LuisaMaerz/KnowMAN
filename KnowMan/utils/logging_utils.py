
def per_step_classifier_tb_logging(log, writer,
                                   tmp_dev_classification_report_dict, tmp_test_classification_report_dict,
                                   total_steps, batches_between_logging, classification_report_dict,
                                   avg_train_losses_classifier, avg_train_losses_domain_blurrer,
                                   avg_train_losses_classifier_dom_blurrer, log_dev_test=True, use_accuracy=True):
    log.info(f'Training loss classifier over the last {batches_between_logging} batches: {avg_train_losses_classifier}')
    log.info(
        f'Training loss domain blurrer over the last {batches_between_logging} '
        f'batches: {avg_train_losses_domain_blurrer}')
    log.info(
        f'Training loss classifier and domain blurrer together over the last {batches_between_logging} '
        f'batches: {avg_train_losses_classifier_dom_blurrer}')
    writer.add_scalar(
        'Training loss classifier accumulated over last batches', avg_train_losses_classifier, total_steps)
    writer.add_scalar(
        'Training loss domain blurrer accumulated over last batches', avg_train_losses_domain_blurrer, total_steps)
    writer.add_scalar(
        'Training loss classifier and domain blurrer together accumulated over last batches',
        avg_train_losses_classifier_dom_blurrer, total_steps)

    if use_accuracy:
        log.info('Classification Training accuracy: {}%'.format(classification_report_dict["accuracy"]))
        writer.add_scalar("Classification accuracy train", classification_report_dict["accuracy"], total_steps)
        if log_dev_test:
            if tmp_dev_classification_report_dict is not None:
                log.info(f'Average validation accuracy: {100.0 * tmp_dev_classification_report_dict["accuracy"]}%')
                writer.add_scalar("Classification accuracy dev", tmp_dev_classification_report_dict["accuracy"],
                                  total_steps)
            log.info(f'Average test accuracy: {100.0 * tmp_test_classification_report_dict["accuracy"]}%')

    else:
        log.info('Classification Training weighted avg f1 score: '
                 '{}%'.format(classification_report_dict['1']["f1-score"]))
        writer.add_scalar("Classification weighted-f1 train", classification_report_dict['1']["f1-score"], total_steps)
        writer.add_scalar("Classification prec (1) train", classification_report_dict['1']["precision"],
                          total_steps)
        writer.add_scalar("Classification recall (1) train", classification_report_dict['1']["recall"],
                          total_steps)
        if log_dev_test:
            log.info(f'Average test weighted-f1: {100.0 * tmp_test_classification_report_dict["1"]["f1-score"]}%')
            if tmp_dev_classification_report_dict is not None:
                log.info(f'Average validation weighted-f1: '
                         f'{100.0 * tmp_dev_classification_report_dict["1"]["f1-score"]}%')
                writer.add_scalar("Classification weighted-f1 dev", tmp_dev_classification_report_dict['1']["f1-score"],
                                  total_steps)
                writer.add_scalar("Classification prec (1) dev",
                                  tmp_dev_classification_report_dict['1']["precision"], total_steps)
                writer.add_scalar("Classification recall (1) dev",
                                  tmp_dev_classification_report_dict['1']["recall"], total_steps)

    return writer


def per_epoch_classifier_tb_logging(writer, classification_report_dict, tmp_dev_classification_report_dict,
                                    tmp_test_classification_report_dict, avg_train_losses_classifier_per_epoch,
                                    avg_train_losses_domain_blurrer_per_epoch,
                                    avg_train_losses_classifier_dom_blurrer_per_epoch, epoch):
    writer.add_scalar("Per epoch Classification accuracy train", classification_report_dict["accuracy"], epoch)
    writer.add_scalar("Per epoch Classification weighted-f1 train", classification_report_dict['1']["f1-score"],
                      epoch)
    writer.add_scalar("Per epoch Classification prec (1) train", classification_report_dict['1']["precision"],
                      epoch)
    writer.add_scalar("Per epoch Classification recall (1) train", classification_report_dict['1']["recall"],
                      epoch)

    writer.add_scalar("Per epoch Classification accuracy dev", tmp_dev_classification_report_dict["accuracy"],
                      epoch)
    writer.add_scalar("Per epoch Classification accuracy test", tmp_test_classification_report_dict["accuracy"],
                      epoch)
    writer.add_scalar("Per epoch Classification weighted-f1 dev",
                      tmp_dev_classification_report_dict['1']["f1-score"],
                      epoch)
    writer.add_scalar("Per epoch Classification weighted-f1 test",
                      tmp_test_classification_report_dict['1']["f1-score"],
                      epoch)
    writer.add_scalar("Per epoch Classification prec (1) dev",
                      tmp_dev_classification_report_dict['1']["precision"], epoch)
    writer.add_scalar("Per epoch Classification prec (1) test",
                      tmp_test_classification_report_dict['1']["precision"], epoch)
    writer.add_scalar("Per epoch Classification recall (1) dev",
                      tmp_dev_classification_report_dict['1']["recall"], epoch)
    writer.add_scalar("Per epoch Classification recall (1) test",
                      tmp_test_classification_report_dict['1']["recall"], epoch)
    writer.add_scalar(
        'Per epoch Training loss classifier accumulated over last batches', avg_train_losses_classifier_per_epoch,
        epoch)
    writer.add_scalar(
        'Per epoch Training loss domain blurrer accumulated over last batches',
        avg_train_losses_domain_blurrer_per_epoch, epoch)
    writer.add_scalar(
        'Per epoch Training loss classifier and domain blurrer together accumulated over last batches',
        avg_train_losses_classifier_dom_blurrer_per_epoch, epoch)
    return writer
