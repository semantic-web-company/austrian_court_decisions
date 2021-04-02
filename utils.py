import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Iterator, Tuple, Dict

import toml
from tqdm import tqdm
# from nltk.corpus import stopwords
import transformers as trs

from nif.annotation import NIFDocument, NIFAnnotation
from HyperBertCLS import prepare_hyperbert_input, WiCTSVDatasetCharOffsets


this_folder = Path(__file__).parent
config = toml.load(this_folder / 'config.toml')


@dataclass
class EntityType:
    tag: str
    hypernyms: List[str]
    definition: str = None


@dataclass
class EntityInContext:
    context: str
    start_ind: int  # in chars
    end_ind: int  # in chars
    target_str: str
    tag: str = None

    def validate_target(self):
        return self.target_str == self.context[self.start_ind:self.end_ind]

    def to_tsv_str(self):
        out = '\t'.join([self.target_str, str(self.start_ind), str(self.end_ind), self.context])
        return out

    def __eq__(self, other):
        if not isinstance(other, EntityInContext):
            return False
        return (self.context == other.context and
                self.start_ind == other.start_ind and self.end_ind == other.end_ind)

    def __hash__(self):
        return f'{self.context}{self.start_ind}{self.end_ind}'.__hash__()


@dataclass
class RecognitionResult:
    entity: EntityInContext
    logits: Dict[str, float]

    def predicted_types(self):
        preds = dict()
        for t, logit in self.logits.items():
            if logit > 0:
                preds[t] = logit
        if not preds:
            preds['Other'] = 1
        return preds


def char_offset2token_offset(
        cxt: str,
        si: int,
        ei: int) -> Tuple[List[str], int, int]:
    cxt_ = cxt[:si] + ' MATCH ' + cxt[ei:]
    target_toks = cxt[si:ei].split()
    tokens = cxt_.split()
    for i, w in enumerate(tokens):
        if w == 'MATCH':
            tgt_si = i
            break
    else:
        raise IndexError(f'No MATCH found in {tokens}')
    tgt_ei = tgt_si + len(target_toks)
    tokens[tgt_si:tgt_si + 1] = target_toks
    return tokens, tgt_si, tgt_ei


def do_disambiguate(
        model: trs.PreTrainedModel,
        tokenizer: trs.PreTrainedTokenizer,
        cxt_entity: EntityInContext,
        types: List[EntityType]):
    tokens, tgt_si, tgt_ei = char_offset2token_offset(cxt=cxt_entity.context,
                                                      si=cxt_entity.start_ind,
                                                      ei=cxt_entity.end_ind)
    encodings = prepare_hyperbert_input(tokenizer,
                                        tokens,
                                        tgt_si,
                                        tgt_ei,
                                        hypernyms_ls=[t.hypernyms for t in types],
                                        definitions=[t.definition for t in types])
    rs = model(**encodings)
    return rs


def nif_corpus2ner_contexts(nif_corpus: Iterable[NIFDocument], tokens_window=350) -> Iterator[EntityInContext]:
    for nif_doc in nif_corpus:
        for ann in nif_doc.annotations:
            surface_form = ann.nif__anchor_of.toPython()
            ann: NIFAnnotation
            au = list(ann.annotation_units.values())[0]
            ner_tag = au.itsrdf__ta_class_ref

            doc_text = nif_doc.context.nif__is_string.toPython()
            start_ind_original = ann.nif__begin_index.toPython()
            cxt_start = start_ind_original-tokens_window if tokens_window < start_ind_original else None
            end_index_original = ann.nif__end_index.toPython()
            cxt_end = end_index_original+tokens_window if tokens_window+end_index_original < len(doc_text) else None
            context = doc_text[cxt_start:cxt_end]
            start_index = tokens_window if cxt_start is not None else start_ind_original
            end_index = start_index + len(surface_form)

            cxt_ent = EntityInContext(
                context=context,
                start_ind=start_index,
                end_ind=end_index,
                target_str=surface_form,
                tag=ner_tag
            )
            assert cxt_ent.validate_target(), (context[start_index:end_index], len(context), start_ind_original,
                                               cxt_start, start_index, end_index_original, cxt_end,
                                               end_index, surface_form)

            yield cxt_ent


def iter_corpus(dataset_folder: Path,
                file_pattern='*.n3',
                doc_limit=None,
                seen_doc_stems=None):
    if seen_doc_stems is None:
        seen_doc_stems = set()
    total_anns = 0
    pbar = tqdm(list(dataset_folder.glob(file_pattern))[:doc_limit])
    for nif_file in pbar:
        nif_file_stem = nif_file.name.split('.')[0]
        if nif_file_stem in seen_doc_stems:
            continue
        with nif_file.open() as f:
            content = f.read()
        n = NIFDocument.parse_rdf(content)
        total_anns += len(n.annotations)
        pbar.set_description(desc=f'Already {total_anns} annotations')
        yield n, nif_file


def prepare_fine_tuning_ds(tokenizer,
                           chosen_classes: Dict[str, List[str]],
                           contexts, start_ends, tags,
                           other_contexts=None, other_start_ends=None,
                           negative_examples_per_instance=3,
                           random_sample_size=300,
                           samples_per_class=None,
                           verbose=False,
                           logger=None):
    chosen_inds = range(len(contexts))
    if random_sample_size is not None:
        chosen_inds = random.sample(chosen_inds, k=random_sample_size)
    elif samples_per_class is not None:
        cls_counter = {cls: [] for cls in chosen_classes}
        chosen_inds = []
        while True:
            i = random.randrange(len(contexts))
            cls_i = tags[i]
            if len(cls_counter[cls_i]) < samples_per_class and i not in chosen_inds:
                cls_counter[cls_i].append(i)
                chosen_inds.append(i)
                if all(len(cls_is) >= samples_per_class for cls_is in cls_counter.values()):
                    break
    sampled_contexts = [contexts[ind] for ind in chosen_inds]
    sampled_start_ends = [start_ends[ind] for ind in chosen_inds]
    sampled_tags = [tags[ind] for ind in chosen_inds]
    sampled_hyps_ls = []
    sampled_defs = []

    if other_contexts is None:
        assert other_start_ends is None
        other_contexts = other_start_ends = other_tags = []
    else:
        assert len(other_contexts) == len(other_start_ends)
        other_tags = [None] * len(other_contexts)
    #
    negative_contexts = []
    negative_start_ends = []
    negative_tags = []
    negative_hyps_ls = []
    negative_defs = []
    for cxt, se, tag in zip(sampled_contexts + other_contexts,
                            sampled_start_ends + other_start_ends,
                            sampled_tags + other_tags):
        if tag is not None:
            sampled_def, *sampled_hyps = chosen_classes[tag]
            sampled_defs.append(sampled_def)
            sampled_hyps_ls.append(sampled_hyps)

        false_tags = set(chosen_classes.keys()) - {tag}
        chosen_false_tags = random.sample(false_tags, k=negative_examples_per_instance)
        for chosen_false_tag in chosen_false_tags:
            false_def, *false_hyps = chosen_classes[chosen_false_tag]
            negative_defs.append(false_def)
            negative_hyps_ls.append(false_hyps)
            negative_contexts.append(cxt)
            negative_start_ends.append(se)
            negative_tags.append(chosen_false_tag)

    ds_cxts = sampled_contexts + negative_contexts
    ds_ses = sampled_start_ends + negative_start_ends
    ds_defs = sampled_defs + negative_defs
    ds_hyps = sampled_hyps_ls + negative_hyps_ls
    ds_labels = [1]*len(sampled_contexts) + [0]*len(negative_contexts)
    ds = WiCTSVDatasetCharOffsets(tok=tokenizer,
                                  contexts=ds_cxts, target_ses=ds_ses,
                                  definitions=ds_defs, hypernyms=ds_hyps,
                                  labels=ds_labels)
    if verbose:
        info_message = f'Dataset size: {len(ds_cxts)}\n' \
                       f'Positive examples tags: {Counter(sampled_tags)}\n' \
                       f'Negative: {Counter(negative_tags)}'
        logger.info(info_message)
    return ds
