import csv
import logging
import random
from collections import defaultdict
from logging import handlers
from pathlib import Path
from typing import List, Dict

import transformers as trs
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from HyperBert.HyperBert3 import HyperBert3 as HyperBERT
import utils

model_path = Path(utils.config['wic_tsv_pretrained_model'])
tokenizer_name = utils.config['tokenizer_name']


def load_model(model_path=model_path, tokenizer_name=tokenizer_name):
    tokenizer = trs.AutoTokenizer.from_pretrained(tokenizer_name)
    model = HyperBERT.from_pretrained(str(model_path))
    return model, tokenizer


def read_cybly_gold_samples(gold_path: Path, logger):
    cxt_ents = []
    tag2type = dict()
    for file_path in gold_path.iterdir():
        if not file_path.is_file() or file_path.suffix != '.tsv':
            logger.info(f'Skipping {file_path}')
            continue
        with file_path.open() as f:
            first_line = f.readline()
            cls_id, def_, hyps, *args = first_line.split('\t')
            ent_type = utils.EntityType(tag=cls_id, definition=def_, hypernyms=hyps.split(', '))
            tag2type[cls_id] = ent_type
            f.readline()
            for l in f.readlines():
                if not l.strip(): continue
                target_str, si, ei, cxt = l.split('\t')
                si = int(si)
                ei = int(ei)
                #
                cxt_ent = utils.EntityInContext(
                    context=cxt,
                    start_ind=si,
                    end_ind=ei,
                    target_str=target_str,
                    tag=cls_id
                )
                assert cxt_ent.validate_target(), (file_path, target_str, cxt[si:ei], si, ei)
                cxt_ents.append(cxt_ent)
            logger.info(f'Read {file_path}. {len(cxt_ents)} instances found.')
    return cxt_ents, tag2type


def fine_tune(out_dir,
              model,
              train_ds,
              dev_ds,
              logger,
              epochs=5):
    def compute_metrics(p) -> Dict:
        binary_preds = (p.predictions > 0).astype(type(p.label_ids[0]))
        acc = accuracy_score(y_true=p.label_ids, y_pred=binary_preds)
        precision, r, f1, _ = precision_recall_fscore_support(y_true=p.label_ids, y_pred=binary_preds, average='binary')
        return {
            "acc": acc,
            "F_1": f1,
            "P": precision,
            "R": r,
            "Positive": binary_preds.sum() / binary_preds.shape[0]
        }

    training_args = trs.TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # evaluate_during_training=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        # eval_steps=10,
        # logging_steps=20,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_F_1',
        greater_is_better=True,
    )
    trainer = trs.Trainer(
        model=model,
        args=training_args,
        # prediction_loss_only=True,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # trainer.save_model()
    logger.info(trainer.log_history)
    try:
        return trainer.model
    finally:
        del trainer


def read_manually_verified(verification_folder: Path):
    correct_classes = defaultdict(list)
    incorrect_classes = defaultdict(list)
    true = []
    predicted = []

    for doc in verification_folder.glob('*.tsv'):
        with open(doc) as csvfile:
            file_name = doc.stem
            tsvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in tsvreader:
                if row[2] == 'True type':
                    continue
                osi = int(row[4])
                oei = int(row[5])
                target = row[3]
                cxt = row[6]
                if cxt[osi:oei] == target:
                    eic = utils.EntityInContext(
                        context=cxt,
                        start_ind=osi,
                        end_ind=oei,
                        target_str=target
                    )
                elif cxt[osi + 1:oei + 1] == target:
                    eic = utils.EntityInContext(
                        context=cxt,
                        start_ind=osi + 1,
                        end_ind=oei + 1,
                        target_str=target
                    )
                elif cxt[osi - 1:oei - 1] == target:
                    eic = utils.EntityInContext(
                        context=cxt,
                        start_ind=osi - 1,
                        end_ind=oei - 1,
                        target_str=target
                    )
                else:
                    assert False, (file_name, cxt[osi:oei], target)
                assert eic.validate_target()
                if row[0] == 'TRUE':
                    eic.tag = file_name
                    correct_classes[file_name].append(eic)
                elif row[0] == 'FALSE':
                    assert row[2], (row, file_name)
                    eic.tag = row[2]
                    incorrect_classes[eic.tag].append(eic)
                else:
                    assert False, (row, file_name)
                true.append(eic.tag)
                predicted.append(file_name)
    return correct_classes, incorrect_classes, true, predicted


def prepare_cybly_train_ds_extension(
        tokenizer,
        incorrect_eics_dict: Dict[str, List[utils.EntityInContext]],
        original_gold_eics: List[utils.EntityInContext],
        tag2type: Dict[str, utils.EntityType],
        logger,
        training_neg_examples_per_positive=5,
        extend_by: int = None,
):
    cxt_ents = original_gold_eics[:]
    other_eics = []
    dev_eics = []
    other_dev_eics = []
    for ent_type, eics in incorrect_eics_dict.items():
        if ent_type == 'Other':
            other_dev_eics = eics[:15]
        else:
            dev_eics += eics[:15]
        #
        if extend_by is not None:
            if len(eics) >= 20:
                info_str = f'Type {ent_type} being extended by {extend_by} instances.'
                print(info_str)
                logger.info(info_str)
                new_eics = random.sample(eics, extend_by)
                if ent_type == 'Other':
                    other_eics = new_eics
                else:
                    cxt_ents += new_eics
        else:
            if ent_type == 'Other':
                other_eics = eics[:100]
            else:
                cxt_ents += eics
    assert other_dev_eics

    train_ds = utils.prepare_fine_tuning_ds(
        tokenizer=tokenizer,
        chosen_classes={k: ([v.definition] + v.hypernyms) for k, v in tag2type.items()},
        contexts=[c.context for c in cxt_ents],
        start_ends=list(zip([c.start_ind for c in cxt_ents], [c.end_ind for c in cxt_ents])),
        tags=[c.tag for c in cxt_ents],
        other_contexts=[c.context for c in other_eics],
        other_start_ends=list(zip([c.start_ind for c in other_eics], [c.end_ind for c in other_eics])),
        negative_examples_per_instance=training_neg_examples_per_positive,
        random_sample_size=None,
        samples_per_class=None,
        verbose=True,
        logger=logger
    )
    dev_ds = utils.prepare_fine_tuning_ds(
        tokenizer=tokenizer,
        chosen_classes={k: ([v.definition] + v.hypernyms) for k, v in tag2type.items()},
        contexts=[c.context for c in dev_eics],
        start_ends=list(zip([c.start_ind for c in dev_eics], [c.end_ind for c in dev_eics])),
        tags=[c.tag for c in dev_eics],
        other_contexts=[c.context for c in other_dev_eics],
        other_start_ends=list(zip([c.start_ind for c in other_dev_eics], [c.end_ind for c in other_dev_eics])),
        negative_examples_per_instance=2,
        random_sample_size=None,
        samples_per_class=None,
        verbose=True,
        logger=logger
    )
    return train_ds, dev_ds


if __name__ == '__main__':
    # random.seed(45)
    manual_annotations_folder = Path(utils.config['manual_annotations_folder'])
    manual_verified_folder = Path(utils.config['manual_verified_folder'])
    output_folder = Path(utils.config['output_folder'])
    log_path = Path(utils.config['log_path'])

    logger = logging.getLogger(__name__)
    handler = handlers.RotatingFileHandler(log_path, maxBytes=1000000, backupCount=3, encoding="UTF-8")
    handler.doRollover()
    handler.setLevel(logging.DEBUG)
    trs.logger.addHandler(handler)
    trs.logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    original_gold_eics, tag2type = read_cybly_gold_samples(gold_path=manual_annotations_folder, logger=logger)
    assert len(tag2type) == 9, tag2type
    correct_classes, incorrect_classes, true, predicted = read_manually_verified(
        verification_folder=manual_verified_folder)
    verified_cxt_ents = []
    for eics in correct_classes.values():
        verified_cxt_ents += eics
    for eics in incorrect_classes.values():
        verified_cxt_ents += eics
    ent_types = list(tag2type.values())
    # Original evaluation
    types_ordering = list(set(true))
    report = classification_report(y_true=true, y_pred=predicted, labels=types_ordering)
    info_str = 'Report original classification'
    print('Report original classification')
    print(report)
    logger.info(info_str)
    logger.info(report)

    model, tokenizer = load_model()
    trains_dss = []
    dev_dss = []
    for i, extension_size in enumerate([10, 10, 10, 5, 5, 5]):
        original_gold_eics, tag2type = read_cybly_gold_samples(gold_path=manual_annotations_folder,
                                                               logger=logger)
        correct_classes, incorrect_classes, _, _ = read_manually_verified(
            verification_folder=manual_verified_folder)
        #
        train_ds, dev_ds = prepare_cybly_train_ds_extension(
            tokenizer=tokenizer,
            extend_by=extension_size,
            incorrect_eics_dict=incorrect_classes,
            original_gold_eics=original_gold_eics,
            training_neg_examples_per_positive=6,
            logger=logger,
            tag2type=tag2type
        )
        trains_dss.append(train_ds)
        dev_dss.append(dev_ds)

    for i, (train_ds, dev_ds) in enumerate(zip(trains_dss, dev_dss)):
        finetuned_model_output_dir = output_folder / f'tuned_model_{i}'
        tuned_model = fine_tune(out_dir=finetuned_model_output_dir,
                                model=model,
                                train_ds=train_ds,
                                dev_ds=dev_ds,
                                epochs=7,
                                logger=logger)
        print('Fine tuning done')
        logger.info('Fine tuning done')
        #
        # preds_per_type = defaultdict(list)
        true = []
        predicted = []
        tuned_model.eval()
        for cxt_ent in verified_cxt_ents:
            rs = utils.do_disambiguate(
                model=tuned_model, tokenizer=tokenizer,
                cxt_entity=cxt_ent,
                types=ent_types
            )
            logits = rs[0].tolist()
            rrs = utils.RecognitionResult(
                entity=cxt_ent,
                logits={ent_types[i].tag: logits[i][0] for i in range(len(ent_types))}
            )
            pred_types = rrs.predicted_types()
            logit2pred = {logit: pred for pred, logit in pred_types.items()}
            best_prediction = logit2pred[max(logit2pred)]
            preds_file = finetuned_model_output_dir / 'predictions.tsv'
            with open(preds_file, 'a') as f:
                pred_str = '\t'.join([cxt_ent.target_str,
                                      str(cxt_ent.start_ind),
                                      str(cxt_ent.end_ind),
                                      cxt_ent.context,
                                      cxt_ent.tag,
                                      best_prediction]) + '\n'
                f.write(pred_str)
            true.append(cxt_ent.tag)
            predicted.append(best_prediction)
        report = classification_report(y_true=true, y_pred=predicted, labels=types_ordering)
        info_str = f'\n\n\nReport for run {i}'
        print(info_str)
        print(report)
        logger.info(info_str)
        logger.info(report)
