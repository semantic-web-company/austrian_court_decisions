from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EvalPrediction
from transformers import Trainer, TrainingArguments


def prepare_hyperbert_input(tok,
                            original_tokens: List[str], original_si: int, original_ei: int,
                            hypernyms_ls: List[List[str]] = None, definitions: List[str] = None,
                            focus_token='$'):
    tokens = original_tokens[:]
    if focus_token is not None:
        tokens.insert(original_ei, focus_token)  # after target
        tokens.insert(original_si, focus_token)  # before target
    context = ' '.join(tokens)

    rows = []
    for hyps, def_ in zip(hypernyms_ls, definitions):
        hyps_str = f' {focus_token} '.join(hyps) if hyps is not None else None
        sense_str = ' ; '.join(x for x in [def_, hyps_str] if x is not None)
        assert any([def_, hyps]), sense_str
        row = [context, sense_str]
        rows.append(row)
    encodings = tok(rows, return_tensors='pt', truncation=True, padding=True)

    return encodings


class HyperBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hyper_classifier = nn.Linear(config.hidden_size, 1)  # BERT
        self.dropout = nn.Dropout(2*config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        cls_output = bert_output[1]  # (bs, dim)

        pooled_output = cls_output

        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.hyper_classifier(pooled_output)  # (bs, 1)

        outputs = (logits,)  # + bert_output[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits,  # (hidden_states), (attentions)


class WiCTSVDataset(torch.utils.data.Dataset):
    def __init__(self, contexts, target_inds, hypernyms, definitions, labels=None, focus_token='$'):
        self.len = len(contexts)
        self.labels = labels
        if focus_token is not None:
            prep_cxts = []
            prep_tgt_inds = []
            for cxt, tgt_ind in zip(contexts, target_inds):
                prep_cxt = cxt.split(' ')
                prep_cxt.insert(tgt_ind + 1, focus_token)  # after target
                prep_cxt.insert(tgt_ind, focus_token)  # before target
                prep_tgt_ind = tgt_ind + 1
                prep_cxts.append(' '.join(prep_cxt))
                prep_tgt_inds.append(prep_tgt_ind)
        else:
            prep_cxts = contexts
            prep_tgt_inds = target_inds
        self.encodings = tok([[context, definition + ' ; ' + f' {focus_token} '.join(hyps)]
                              for context, tgt_ind, definition, hyps in zip(prep_cxts, prep_tgt_inds, definitions, hypernyms)],
                             return_tensors='pt', truncation=True, padding=True)
        real_targets = [cxt.split(' ')[tgt_ind] for cxt, tgt_ind in zip(contexts, target_inds)]
        assert all(rt == p_cxt.split(' ')[p_ind] for rt, p_cxt, p_ind in zip(real_targets, prep_cxts, prep_tgt_inds))

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(float(self.labels[idx]))
        return item

    def __len__(self):
        return self.len


class WiCTSVDatasetCharOffsets(torch.utils.data.Dataset):
    def __init__(self, tok, contexts, target_ses, hypernyms, definitions, labels=None, focus_token='$'):
        self.len = len(contexts)
        self.labels = labels
        if focus_token is not None:
            prep_cxts = []
            for cxt, (tgt_si, tgt_ei) in zip(contexts, target_ses):
                prep_cxt = cxt[:tgt_si] + f' {focus_token} ' + cxt[tgt_si:tgt_ei] + f' {focus_token} ' + cxt[tgt_ei:]
                # prep_cxt.insert(tgt_si + 1, f' {focus_token} ')  # after target
                # prep_cxt.insert(tgt_ei, f' {focus_token} ')  # before target
                assert prep_cxt[tgt_si + 3:tgt_ei + 3] == cxt[tgt_si:tgt_ei]
                prep_cxts.append(prep_cxt)
        else:
            prep_cxts = contexts
        self.encodings = tok([[context, definition + ' ; ' + f' {focus_token} '.join(hyps)]
                              for context, definition, hyps in zip(prep_cxts, definitions, hypernyms)],
                             return_tensors='pt', truncation=True, padding=True)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(float(self.labels[idx]))
        return item

    def __len__(self):
        return self.len


def compute_metrics(p: EvalPrediction) -> Dict:
    binary_preds = (p.predictions > 0).astype(type(p.label_ids[0]))
    preds: np.ndarray
    acc = accuracy_score(y_true=p.label_ids, y_pred=binary_preds)
    precision, r, f1, _ = precision_recall_fscore_support(y_true=p.label_ids, y_pred=binary_preds, average='binary')
    return {
        "acc": acc,
        "F_1": f1,
        "P": precision,
        "R": r,
        "Positive": binary_preds.sum() / binary_preds.shape[0]
    }


def read_wic_tsv_ds(folder_path: Path):
    contexts = []
    target_inds = []
    examples_path = next(folder_path.glob('*examples.txt'))
    with examples_path.open() as ex_f:
        ex_f_lines = ex_f.readlines()
        for line in ex_f_lines:
            _, target_ind, context = line.split('\t')
            target_ind = int(target_ind.strip())
            clean_cxt = context.strip()
            contexts.append(clean_cxt)
            target_inds.append(target_ind)
    hypernyms_path = next(folder_path.glob('*hypernyms.txt'))
    with hypernyms_path.open() as hyp_f:
        hypernyms = [[hyp.replace('_', ' ').strip() for hyp in line.split('\t')]
                     for line in hyp_f.readlines()]
    defs_path = next(folder_path.glob('*definitions.txt'))
    with defs_path.open() as defs_f:
        definitions = defs_f.readlines()
    try:
        labels_path = next(folder_path.glob('*labels.txt'))
        with labels_path.open() as labels_f:
            labels = [x.strip() == 'T' for x in labels_f.readlines()]
    except:
        labels = None
    assert len(contexts) == len(hypernyms) == len(definitions), \
        (len(contexts), len(hypernyms), len(definitions))
    return contexts, target_inds, hypernyms, definitions, labels


if __name__ == '__main__':
    import utils
    import logging
    logging.basicConfig(level=logging.INFO)

    base_path = Path(utils.config['wic_tsv_de_folder'])
    wic_tsv_train = base_path / 'Training'
    wic_tsv_dev = base_path / 'Development'
    output_path = Path(utils.config['wic_tsv_pretrained_model'])

    model_name = utils.config['tokenizer_name']
    tok = AutoTokenizer.from_pretrained(model_name)
    model = HyperBERT.from_pretrained(model_name)

    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv_ds(wic_tsv_train)
    pos_sample_weight = (len(labels) - sum(labels)) / sum(labels)
    print('train', pos_sample_weight, Counter(labels))
    train_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions, labels)

    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv_ds(wic_tsv_dev)
    pos_sample_weight = (len(labels) - sum(labels)) / sum(labels)
    print('dev', pos_sample_weight, Counter(labels))
    dev_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions, labels)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_steps=10,
        logging_steps=10,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_F_1',
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    output = trainer.train()
    print(f'Training output: {output}')
    trainer.save_model()
    # preds = trainer.predict(test_dataset=test_ds)
    # print(preds)
    # print(preds.predictions.tolist())
