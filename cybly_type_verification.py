import logging
from logging import handlers
from pathlib import Path

import transformers as trs

import utils
from evalute_on_manually_verified import read_cybly_gold_samples, read_manually_verified, load_model, \
    prepare_cybly_train_ds_extension, fine_tune


if __name__ == '__main__':
    model_path = Path(utils.config['wic_tsv_pretrained_model'])
    tokenizer_name = utils.config['tokenizer_name']
    nif_annotations_folder = Path(utils.config['nif_annotations_folder'])
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
    ent_types = list(tag2type.values())
    assert len(ent_types) == 9
    finetuned_model_output_dir = output_folder / f'tuned_model_all'
    try:
        tuned_model, tokenizer = load_model(finetuned_model_output_dir)
    except:
        correct_classes, incorrect_classes, _, _ = read_manually_verified(
            verification_folder=manual_verified_folder)
        model, tokenizer = load_model()
        train_ds, dev_ds = prepare_cybly_train_ds_extension(
            tokenizer=tokenizer,
            incorrect_eics_dict=incorrect_classes,
            original_gold_eics=original_gold_eics,
            tag2type=tag2type,
            training_neg_examples_per_positive=6,
            logger=logger,
            extend_by=None
        )
        tuned_model = fine_tune(out_dir=finetuned_model_output_dir,
                                model=model,
                                train_ds=train_ds,
                                dev_ds=dev_ds,
                                logger=logger,
                                epochs=10)
        print('Fine tuning done')
        logger.info('Fine tuning done')
    #
    preds_file = finetuned_model_output_dir / 'predictions.tsv'
    if preds_file.exists():
        with open(preds_file) as f:
            lines = f.readlines()
        seen_doc_stems = {line.split('\t')[6].split('.')[0] for line in lines}
        info = f'Already seen {len(seen_doc_stems)} files'
        print(info)
        logger.info(info)
    else:
        seen_doc_stems = None
        with open(preds_file, 'w') as f:
            pred_str = '\t'.join(['Target string',
                                  'Start offset in context',
                                  'End offset in context',
                                  'Context',
                                  'Original coarse tag',
                                  'Predicted fine tag',
                                  'Original Document',
                                  'Target offset in original document']) + '\n'
            f.write(pred_str)
        info = f'Starting verification from scratch'
        print(info)
        logger.info(info)
    #
    nif_corpus = utils.iter_corpus(nif_annotations_folder, doc_limit=None, seen_doc_stems=seen_doc_stems)
    for nif_doc, nif_file in nif_corpus:
        nif_file_stem = nif_file.stem.split('.')[0]
        eic_iter = utils.nif_corpus2ner_contexts([nif_doc], tokens_window=200)
        for eic in eic_iter:
            target = eic.target_str
            cxt = eic.context
            se = (eic.start_ind, eic.end_ind)
            tag = eic.tag
            doc_offset = nif_doc.context.nif__is_string.find(cxt) + se[0]
            cxt_ent = utils.EntityInContext(context=cxt,
                                            start_ind=se[0],
                                            end_ind=se[1],
                                            target_str=target,
                                            tag=tag)
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
            with open(preds_file, 'a') as f:
                pred_str = '\t'.join([cxt_ent.target_str,
                                      str(cxt_ent.start_ind),
                                      str(cxt_ent.end_ind),
                                      cxt_ent.context,
                                      cxt_ent.tag,
                                      best_prediction,
                                      nif_file_stem + '.de.json',
                                      str(doc_offset)]) + '\n'
                f.write(pred_str)
