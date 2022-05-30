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
