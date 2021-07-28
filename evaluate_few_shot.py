import logging
import random
from logging import handlers
from pathlib import Path

import transformers as trs
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from HyperBert.HyperBert3 import HyperBert3
import utils
import evalute_on_manually_verified as eval_man
from model_evaluation.wictsv_dataset import WiCTSVDatasetCharOffsets

model_path = Path(utils.config['wic_tsv_pretrained_model3'])
tokenizer_name = utils.config['tokenizer_name']


def load_model(model_path=model_path, tokenizer_name=tokenizer_name):
    tokenizer = trs.AutoTokenizer.from_pretrained(tokenizer_name)
    model = HyperBert3.from_pretrained(str(model_path))
    return model, tokenizer


def read_conll_with_targets(conll_path, add_negative_neighbors=True):
    conll_path = Path(conll_path)
    cxts = []
    ses = []
    type_labels = []
    other_cxts = []
    other_ses = []
    other_type_labels = []
    with open(conll_path) as f_reader:
        lines = f_reader.readlines()
        current_sent = []
        si = None
        ei = None
        for line in lines:
            if not line.strip():
                if current_sent:
                    assert target_type_label and target_type_label != 'None', (conll_path, current_sent, target_type_label)
                    char_si = sum(len(word) + 1 for word in current_sent[:si])
                    char_ei = sum(len(word) + 1 for word in current_sent[:ei]) - 1
                    if target_type_label == 'O':
                        other_ses.append((char_si, char_ei))
                        other_cxts.append(' '.join(current_sent))
                        other_type_labels.append(target_type_label)
                    else:
                        cxts.append(' '.join(current_sent))
                        ses.append((char_si, char_ei))
                        type_labels.append(target_type_label)
                        if add_negative_neighbors:
                            if si > 0:
                                prev_word_char_si = sum(len(word) + 1 for word in current_sent[:si-1])
                                prev_word_char_ei = sum(len(word) + 1 for word in current_sent[:si]) - 1
                                other_cxts.append(' '.join(current_sent))
                                other_ses.append((prev_word_char_si, prev_word_char_ei))
                                other_type_labels.append('O')
                            if ei <= len(current_sent):
                                next_word_char_si = sum(len(word) + 1 for word in current_sent[:ei])
                                next_word_char_ei = sum(len(word) + 1 for word in current_sent[:ei+1]) - 1
                                other_cxts.append(' '.join(current_sent))
                                other_ses.append((next_word_char_si, next_word_char_ei))
                                other_type_labels.append('O')
                    current_sent = []
                    si = None
                    ei = None
                    target_type_label = None
            else:
                word, label = tuple(map(lambda s: s.strip(), line.split('\t')))
                if label != 'None':
                    if si is None:
                        si = len(current_sent)
                        target_type_label = label.split('-')[-1].split(',')[0]
                else:
                    if si is not None and ei is None:
                        ei = len(current_sent)
                current_sent.append(word)
    return cxts, ses, type_labels, other_cxts, other_ses, other_type_labels


def evaluate_few_shot(model, tokenizer,
                      tag2type,
                      shots_path, dev_path, test_path, output_folder,
                      logger):
    ent_types = list(tag2type.values())
    chosen_types_dict = {k.split(",")[0]: ([v.definition] + v.hypernyms) for k, v in tag2type.items()}

    # dev set
    dev_cxts, dev_ses, dev_type_labels, dev_other_cxts, dev_other_ses, dev_other_type_labels = read_conll_with_targets(
        dev_path, add_negative_neighbors=False)
    assert set(dev_other_type_labels) == {'O'}
    dev_ds = utils.prepare_fine_tuning_ds(
        tokenizer=tokenizer,
        chosen_classes=chosen_types_dict,
        contexts=dev_cxts,
        start_ends=dev_ses,
        tags=dev_type_labels,
        other_contexts=dev_other_cxts,
        other_start_ends=dev_other_ses,
        #
        negative_examples_per_instance=4,
        random_sample_size=None,
        samples_per_class=None,
        verbose=True,
        logger=logger
    )
    # test set
    test_cxts, test_ses, test_type_labels, test_other_cxts, test_other_ses, test_other_type_labels = read_conll_with_targets(
        test_path, add_negative_neighbors=False)
    test_cxts += test_other_cxts
    test_ses += test_other_ses
    test_type_labels += test_other_type_labels

    all_hyps = []
    all_defs = []
    for ent_type in ent_types:
        this_type_hyps = [ent_type.hypernyms] * len(test_cxts)
        all_hyps += this_type_hyps
        this_type_defs = [ent_type.definition] * len(test_cxts)
        all_defs += this_type_defs
    test_ds_cxts = test_cxts * len(ent_types)
    test_ds_ses = test_ses * len(ent_types)
    test_ds_labels = []
    for x in test_type_labels:
        for ent_type in ent_types:
            test_ds_labels.append(x.split('-')[-1] == ent_type.tag.split(',')[0])
    test_ds = WiCTSVDatasetCharOffsets(tokenizer=tokenizer,
                                       contexts=test_ds_cxts,
                                       target_ses=test_ds_ses,
                                       definitions=all_defs,
                                       hypernyms=all_hyps,
                                       labels=test_ds_labels
                                       )
    # train sets
    train_dss = []
    for train_shot_filename in shots_path.iterdir():
        train_cxts, train_ses, train_type_labels, train_other_cxts, train_other_ses, train_other_type_labels = read_conll_with_targets(
            train_shot_filename)
        train_ds = utils.prepare_fine_tuning_ds(
            tokenizer=tokenizer,
            chosen_classes=chosen_types_dict,
            contexts=train_cxts,
            start_ends=train_ses,
            tags=train_type_labels,
            other_contexts=train_other_cxts,
            other_start_ends=train_other_ses,
            negative_examples_per_instance=6,
            random_sample_size=None,
            samples_per_class=None,
            verbose=True,
            logger=logger
        )
        train_dss.append(train_ds)
    # fine tune and evaluate
    preds_file = Path(output_folder) / 'predictions.tsv'
    for i, train_ds in enumerate(train_dss):
        finetuned_model_output_dir = output_folder / f'tuned_model_{i}'
        tuned_model = eval_man.fine_tune(out_dir=finetuned_model_output_dir,
                                         model=model,
                                         train_ds=train_ds,
                                         dev_ds=dev_ds,
                                         epochs=8,
                                         logger=logger)
        print('Fine tuning done')
        logger.info('Fine tuning done')

        # tuned_model = model

        trainer = trs.Trainer(model=tuned_model)
        pred_logits, pred_labels, loss = trainer.predict(test_dataset=test_ds)
        true = []
        predicted = []
        for j in range(len(test_cxts)):
            preds_j = [pred_logits[j + k * len(test_cxts)][0] for k in range(len(ent_types))]
            rrs = utils.RecognitionResult(
                entity=None,
                logits={ent_types[i].tag.split(',')[0]: preds_j[i] for i in range(len(ent_types))}
            )
            real_tag = test_type_labels[j].split('-')[-1]
            pred_types = rrs.predicted_types('O')
            logit2pred = {logit: pred for pred, logit in pred_types.items()}
            best_prediction = logit2pred[max(logit2pred)]
            with open(preds_file, 'a') as f:
                pred_str = '\t'.join([test_cxts[j][test_ses[j][0]:test_ses[j][1]],
                                      str(test_ses[j][0]),
                                      str(test_ses[j][1]),
                                      test_cxts[j],
                                      real_tag,
                                      best_prediction]) + '\n'
                f.write(pred_str)
            true.append(real_tag)
            predicted.append(best_prediction)
        report = classification_report(y_true=true, y_pred=predicted)
        info_str = f'\n\n\nReport for run {i}'
        print(info_str)
        print(report)
        logger.info(info_str)
        logger.info(report)


if __name__ == '__main__':
    random.seed(45)
    manual_annotations_folder = Path(utils.config['manual_annotations_folder'])
    test_path = Path(utils.config['conll_test_path'])
    dev_path = Path(utils.config['conll_dev_path'])
    path_1shot = Path(utils.config['verified_1shot_folder'])
    path_5shot = Path(utils.config['verified_5shot_folder'])
    output_folder = Path(utils.config['few_shot_hyperbert_output'])
    log_path = Path(utils.config['few_shot_log_path'])

    logger = logging.getLogger(__name__)
    handler = handlers.RotatingFileHandler(log_path, maxBytes=1000000, backupCount=3, encoding="UTF-8")
    handler.doRollover()
    handler.setLevel(logging.DEBUG)
    trs.logger.addHandler(handler)
    trs.logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Load wic-tsv pre-trained model
    model, tokenizer = load_model()
    # the information about types -- definitions and hypernyms -- are loaded from the original annotations
    _, tag2type = eval_man.read_cybly_gold_samples(gold_path=manual_annotations_folder, logger=logger)
    assert len(tag2type) == 9, tag2type

    evaluate_few_shot(model=model, tokenizer=tokenizer,
                      tag2type=tag2type,
                      shots_path=path_5shot,
                      dev_path=dev_path, test_path=test_path, output_folder=output_folder,
                      logger=logger)
