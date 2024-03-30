## Reproducibility

Much like how different tokenization schemes can result in slightly
different values for, e.g., BLEU, slightly different image
preprocessing can result in slightly different CLIPScore values. More
details are available
[here](https://github.com/jmhessel/clipscore/blob/main/README.md#reproducibility-notes). The
official version of CLIPScore is computed in float16 on GPU, and a
warning will be raised if you're using CPU (though the results should
be similar up to floating point precision). Another important
standarization is the particular image files --- because jpg
compression is lossy, small changes (e.g., different resizing, saving
jpgs multiple times, etc.) of images may result in slighly different
CLIPScores. This repo contains precomputed MSCOCO features extracted
on GPU from the 2017 train/val/test images (available
[here](https://cocodataset.org/#download)). For reproduability, you
can compare your MSCOCO images to [these
checksums](https://storage.googleapis.com/ai2-jack-public/clipscore/mscoco_checksum.txt.zip),
or just use the precomputed features on GPU.

## Example usage


A minimal usage example from Jungo Kasai:
```
from pycocoevalcap.eval import ClipScore

scorer = ClipScore()
references = {'000000000974': ["An elephant with a man and three children on its back drinking water in the jungle.",
	     		       "A man riding on and elephants neck and guiding it while children ride on a seat behind him.",
			       "A group of people ride atop of an elephant.",
			       "A guide and three people riding an elephant.",
			       "A few people riding on an elephants back."]}
candidates = {'000000000974': ['A park bench sitting in the middle of a river.']}
average_score, per_caption_scores = scorer.compute_score(references, candidates)
print(per_caption_scores)
```

for float16 GPU, the output expected for this case is:
```
[{'CLIPScore': 0.5474, 'RefCLIPScore': 0.5806}]
```

for float32 CPU (the unofficial run setting, please use GPU if possible!):
```
[{'CLIPScore': 0.54759747, 'RefCLIPScore': 0.58061004}]
```
