import logging
import random
from collections import defaultdict
from logging import handlers
from pathlib import Path

import utils
from evalute_on_manually_verified import read_cybly_gold_samples, read_manually_verified

if __name__ == '__main__':
    manual_verified_folder = Path(utils.config['manual_verified_folder'])
    output_folder = Path(utils.config['output_folder'])
    log_path = Path(utils.config['log_path'])

    logger = logging.getLogger(__name__)
    handler = handlers.RotatingFileHandler(log_path, maxBytes=1000000, backupCount=3, encoding="UTF-8")
    handler.doRollover()
    handler.setLevel(logging.DEBUG)
    # trs.logger.addHandler(handler)
    # trs.logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    random.seed(200)
    correct_classes, incorrect_classes, _, _ = read_manually_verified(
        verification_folder=manual_verified_folder)
    original_verified_eics = sum(correct_classes.values(), []) + sum(incorrect_classes.values(), [])
    # split type tags
    tag2ents = defaultdict(list)
    for eic in original_verified_eics:
        eic.tag = eic.tag.split(',')[0]
        tag2ents[eic.tag].append(eic)
    # entity types
    with open(utils.config['ner_labels_path'], 'w') as f:
        labels = []
        for tag in tag2ents:
            labels.append(f'I-{tag}')
            labels.append(f'B-{tag}')
        f.write('\n'.join(labels))
        f.write('\nO')
        f.write('\nNone')
    # test set -- all entities
    with open(utils.config['conll_test_path'], 'w') as f:
        for eic in original_verified_eics:
            eic.tag = eic.tag.split(',')[0]
            conll_tuples = eic.to_conll(context_window_size=40)
            assert any(x[1] != 'None' for x in conll_tuples), (eic, conll_tuples)
            conll_lines = '\n'.join("\t".join(x) for x in conll_tuples)
            f.write(conll_lines)
            f.write('\n\n')
    # dev set
    with open(utils.config['conll_dev_path'], 'w') as f:
        for tag, ents in tag2ents.items():
            random.shuffle(ents)
            if len(ents) > 20:
                tag_dev_set = ents[:10]
                tag2ents[tag] = ents[10:]
            elif 20 >= len(ents) > 10:
                tag_dev_set = ents[:5]
                tag2ents[tag] = ents[5:]
            else:
                continue
            for ent in tag_dev_set:
                conll_lines = '\n'.join("\t".join(x) for x in ent.to_conll(context_window_size=40))
                f.write(conll_lines)
                f.write('\n\n')
    # 1 shots
    for i in range(5):
        shot_file = Path(utils.config['verified_1shot_folder']) / f'{i}.txt'
        with open(shot_file, 'w') as f:
            for tag, ents in tag2ents.items():
                if tag != 'Other':
                    tag_1ent = random.sample(ents, 1)[0]
                    conll_lines = '\n'.join("\t".join(x) for x in tag_1ent.to_conll(context_window_size=40))
                    f.write(conll_lines)
                    f.write('\n\n')
    # 5 shots
    for i in range(5):
        shot_file = Path(utils.config['verified_5shot_folder']) / f'{i}.txt'
        with open(shot_file, 'w') as f:
            for tag, ents in tag2ents.items():
                if tag != 'Other':
                    tag_5ents = random.sample(ents, 5) if len(ents) > 5 else ents
                    for ent in tag_5ents:
                        conll_lines = '\n'.join("\t".join(x) for x in ent.to_conll(context_window_size=40))
                        f.write(conll_lines)
                        f.write('\n\n')
