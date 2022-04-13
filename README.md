<h1 align="center">
<p>Neural Spelling Correction Toolkit
</h1>

# Contents

- [Installation & Quick Start](#Installation)
- Toolkit
    - [Introduction](#Introduction)
    - [Download Checkpoints](#Download-Checkpoints)
    - [Download Datasets](#Datasets)
# Installation

```bash
git clone https://github.com/melnikoff-oleg/jb-spellchecker; cd jb-spellchecker
pip install -r requirements.txt
```

Here is a quick-start code snippet to use a checker model.

```python
from model.spellcheck_model import BartChecker

""" select spell checkers & load """
checker = BartChecker()
checker.from_pretrained()

""" spell correction """
checker.correct("I luk foward to receving your reply")
# → "I look forward to receiving your reply"
checker.correct_strings(["I luk foward to receving your reply", ])
# → ["I look forward to receiving your reply"]

""" evaluation of models """
checker.evaluate(clean_file="datasets/bea/bea500.gt", corrupt_file="datasets/bea/bea500.noise")
# → "Metrics": {
# →         "Precision": 0.82,
# →         "Recall": 0.91,
# →         "F_0_5": 0.84,
# →         "Word-level accuracy": 0.98
# →     }
```

# Toolkit

### Introduction

This is an open-source toolkit for context sensitive spelling correction in English. This toolkit comprises of 3
spell checkers, with evaluations on naturally occurring mis-spellings from multiple (publicly available) sources. To
make neural models for spell checking context dependent, (i) we train neural models using spelling errors in context,
synthetically constructed by reverse engineering isolated mis-spellings; and  (ii) use richer representations of the
context.This toolkit enables NLP practitioners to use our proposed spelling correction systems via
simple interface.


##### List of neural models in the toolkit:

- [```BART end-to-end```](https://drive.google.com/file/d/14XiDY4BJ144fVGE2cfWfwyjnMwBcwhNa/view?usp=sharing)
- [```BERT-detector + BART-rewriter```](https://drive.google.com/file/d/1OvbkdBXawnefQF1d-tUrd9lxiAH1ULtr/view?usp=sharing)
- [```Char-Based transformer end-to-end```](https://drive.google.com/file/d/19ZhWvBaZqrsP5cGqBJdFPtufdyBqQprI/view?usp=sharing)

<p align="center">
    <br>
    <img src="https://github.com/melnikoff-oleg/jb-spellchecker/blob/main/images/bert-bart-model.png?raw=true" width="400"/>
    <br>
    This pipeline corresponds to the <i>BERT-detector + BART-rewriter</i> model.
<p>

##### Performances

| Spell<br>Checker    | Word<br>Correction <br>Rate | Time per<br>sentence <br>(in milliseconds) |
|-------------------------------------|-----------------------|--------------------------------------|
| ```Aspell```                        | 49%                  | 7.3*                                 |
| ``` Jamspell```                     | 69%                | 2.6*                                 |
| ```BART end-to-end```                      | 90%                  | 6.2                                  |
| ```BERT-detector + BART-rewriter```                       | 90%                  | 10.2                                  |
| ```Char-Based transformer end-to-end```                   | 79%                  | 3.4                                  |

Performance of different correctors in our toolkit on the  ```BEA-60K```  dataset with real-world spelling
mistakes. ∗ indicates evaluation on a CPU (for others we use NVIDIA V100 Tensor Core).

### Download Checkpoints

All models checkpoints are already uploaded to HuggingFace. Every Checker class implements method .from_pretrained() so you can easily download them.
Checkpoints sizes are shown in the table.

| Spell Checker                       | Class               | HuggingFace checkpoint name             | Disk space (approx.) |
|-------------------------------------|---------------------|-----------------------------|----------------------|
| ```BART end-to-end```                      | `BartChecker`    | `melnikoff-oleg/bart-end-to-end`    | 450 MB               |
| ```BERT-detector + BART-rewriter```                       | `BertBartChecker`     | `melnikoff-oleg/bert-bart`       | 700 MB               |
| ```Char-based transformer end-to-end```                   | `CharBasedTransformerChecker` | `melnikoff-oleg/char-based-end-to-end`   | 120 MB               |



### Datasets

We use several synthetic and natural datasets for training/evaluating neural models. Run the following to download all the datasets.

```
cd datasets
python download_datafiles.py
```

See ```datasets/README.md``` for more details.
