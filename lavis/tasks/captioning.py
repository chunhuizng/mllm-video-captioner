"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            try:
                results.append({"caption": caption, "image_id": int(img_id)}) ### coco caption
            except:
                results.append({"caption": caption, "image_id": img_id}) ### msrvtt caption

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val["cider"] + coco_val["bleu"][-1]
        log_stats = {split_name: {k: v for k, v in coco_val.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

# Add support to evaluate custom captioning datasets
# pip install git+https://github.com/jmhessel/pycocoevalcap.git
# Based on https://github.com/jmhessel/clipscore
from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.spice.spice import Spice
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.rouge.rouge import Rouge
from torchvision.datasets.utils import download_url


def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'bleu'),
                               (Meteor(), 'meteor'),
                               (Rouge(), 'rouge'),
                               (Cider(), 'cider'),
                               ]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    refs, cands = tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


def caption_eval(coco_gt_root, results_file, split):
    if "coco" in results_file:
        urls = {
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
        }
        filenames = {
            "val": "coco_karpathy_val_gt.json",
            "test": "coco_karpathy_test_gt.json",
        }
        download_url(urls[split], coco_gt_root)
        annotation_file = os.path.join(coco_gt_root, filenames[split])

    elif "msrvtt" in results_file:
        filenames = {
            "val": "msrvtt_val_gt.json",
            "test": "msrvtt_test_gt.json",
        }
        msrvtt_gt_root = coco_gt_root.replace("coco_gt", "msrvtt_gt")
        annotation_file = os.path.join(msrvtt_gt_root, filenames[split])

    elif "msvd" in results_file:
        filenames = {
            "val": "msvd_val_gt.json",
            "test": "msvd_test_gt.json",
        }
        msrvtt_gt_root = coco_gt_root.replace("coco_gt", "msvd_gt")
        annotation_file = os.path.join(msrvtt_gt_root, filenames[split])
    
    elif "vatex" in results_file:
        filenames = {
            "val": "vatex_val_gt.json",
            "test": "vatex_test_gt.json",
        }
        msrvtt_gt_root = coco_gt_root.replace("coco_gt", "vatex_gt")
        annotation_file = os.path.join(msrvtt_gt_root, filenames[split])

    else:
        print(" ***** Output directory of captioning at LAVIS/lavis/outputs/ has to be name using one of ['Caption_coco', 'Caption_msrvtt', 'Caption_msvd', 'Caption_vatex']. ***** ")
        exit()

    with open(results_file) as f:
        results = json.load(f)

    with open(annotation_file) as f:
        annotations = json.load(f)

    image_ids = []
    candidates = {}
    for res in results:
        candidates[res["image_id"]] = res["caption"]
        image_ids.append(res["image_id"])
    
    references = {}
    for ann in annotations["annotations"]:   # "coco_gt/coco_karpathy_test_gt.json"
        if ann["image_id"] not in references:   # TODO: more sufficient
            references[ann["image_id"]] = []
        references[ann["image_id"]].append(ann["caption"])

    candidates = [candidates[cid] for cid in image_ids]
    references = [references[cid] for cid in image_ids]
    if isinstance(references[0], str):
        references = [[r] for r in references]

    metrics = get_all_metrics(references, candidates)
    for k, v in metrics.items():
        if k == 'bleu':
            for bidx, sc in enumerate(v):
                print('BLEU-{}: {:.4f}'.format(bidx+1, sc))
        else:
            print('{}: {:.4f}'.format(k.upper(), v))

    return metrics
