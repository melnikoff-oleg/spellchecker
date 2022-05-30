<h1 align="center">
<p>Models for correcting typos
</h1>

# Contents

- [Installation & Quick Start](#Installation)
- Toolkit
    - [Introduction](#Introduction)
    - [Download Checkpoints](#Download-Checkpoints)
    - [Download Datasets](#Datasets)
# Installation

```bash
git clone https://github.com/melnikoff-oleg/spellchecker; cd spellchecker
pip install -r requirements.txt
```

Here is a quick-start code snippet to use a checker model.

```python
from model.spellcheck_model import BartChecker, BartSepMaskAllChecker, DCR
from evaluation.evaluate import evaluate
from data_utils.utils import get_texts_from_file

""" select spell checkers & load """
checker = BartSepMaskAllChecker()
checker.from_pretrained()

""" spell correction """
print(checker.correct("I luk foward to receving your reply"))
# → "I look forward to receiving your reply"
print(checker.correct_strings(["I luk foward to receving your reply", ]))
# → ["I look forward to receiving your reply"]

""" evaluation of models """
texts_gt, texts_noise = get_texts_from_file('dataset/bea/bea500.gt'), get_texts_from_file('dataset/bea/bea500.noise')
evaluate(model=checker, texts_gt=texts_gt, texts_noise=texts_noise, exp_save_dir='experiments/bart-sep-mask-all/')
# →     {
# →         "Precision": 0.86,
# →         "Recall": 0.94,
# →         "F_0_5": 0.87,
# →         "Word-level accuracy": 0.99,
# →         "Broken tokenization cases": 0.0,
# →     }
```

# Toolkit

### Introduction

This is an open-source toolkit for context-sensitive spelling correction in English. This toolkit comprises 3
spelling checkers, with evaluations on naturally occurring misspellings from publicly available sources. To
make neural models for spell-checking context dependent, (i) we train neural models using spelling errors in context,
synthetically constructed by reverse engineering isolated mis-spellings; and  (ii) use richer representations of the
context.This toolkit enables NLP practitioners to use our proposed spelling correction systems via
simple interface.


##### List of neural models in the toolkit:

- ```BART end-to-end```
- ```BART Sep-Mask-All```
- ```Detector Candidator Ranker```

<p align="center">
    <br>
    <img src="https://github.com/melnikoff-oleg/jb-spellchecker/blob/main/images/bart-sep-mask-all.png?raw=true" width="800"/>
    <br>
    This pipeline corresponds to the <i>BART Sep-Mask-All</i> model.
<p>

##### Performances

| Spell<br>Checker    | Word<br>Correction <br>Rate | Time per<br>sentence <br>(in milliseconds) |
|-------------------------------------|-----------------------|--------------------------------------|
| ```Aspell```                        | 49%                  | 7.3*                                 |
| ``` Jamspell```                     | 69%                | 2.6*                                 |
| ```BART end-to-end```                      | 85%                  | 396                                  |
| ```BART Sep-Mask-All```                       | 91%                  | 205                                  |
| ```Detector Candidator Ranker```                   | 92%                  | 213                                  |

Performance of different correctors in our toolkit on the  ```BEA-60K```  dataset with real-world spelling
mistakes. ∗ indicates evaluation on a CPU (for others we use NVIDIA V100).

### Download Checkpoints

All models checkpoints are already uploaded to HuggingFace. Every Checker class implements method .from_pretrained() so you can easily download them.
Checkpoints sizes are shown in the table.

| Spell Checker                       | Class               | HuggingFace checkpoint name             | Disk space (approx.) |
|-------------------------------------|---------------------|-----------------------------|----------------------|
| ```BART end-to-end```                      | `BartChecker`    | `melnikoff-oleg/bart-end-to-end`    | 532 MB               |
| ```BART Sep-Mask-All```                       | `BartSepMaskAllChecker`     | `melnikoff-oleg/distilbart-sep-mask-all`       | 424 MB               |
| ```Detector Candidator Ranker```                   | `DCR` | `melnikoff-oleg/distilbart-sep-mask-all`   | 424 MB               |



### Datasets

We use several synthetic and natural datasets for training/evaluating neural models. Run the following to download all the datasets.

```
cd datasets
python download_datafiles.py
```

See ```datasets/README.md``` for more details.
