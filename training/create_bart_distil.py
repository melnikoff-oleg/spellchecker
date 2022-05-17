# https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/scripts/extract.py
import torch
from transformers import BartForConditionalGeneration
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


if __name__ == "__main__":

    # init_checkpoint = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent_v0_214056.pt'
    save_checkpoint = PATH_PREFIX + 'training/checkpoints/oldbart-en03dec03.pt'

    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # print(model.config)
    # config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
    # model = BartForConditionalGeneration(config)
    # model.load_state_dict(torch.load(init_checkpoint))
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    state_dict = model.state_dict()
    compressed_sd = {}

    bad_layers = ['decoder.layers.1.', 'decoder.layers.2.', 'decoder.layers.4.', 'decoder.layers.5.',
                  'encoder.layers.1.', 'encoder.layers.2.', 'encoder.layers.4.', 'encoder.layers.5.']
    mapping = {'decoder.layers.3.': 'decoder.layers.1.',
               'encoder.layers.3.': 'encoder.layers.1.'}

    for param_name in state_dict.keys():
        ok = True
        for bl in bad_layers:
            if bl in param_name:
                ok = False
        if ok:
            final_param_name = param_name
            for map_key in mapping.keys():
                if map_key in param_name:
                    final_param_name = param_name.replace(map_key, mapping[map_key])
            compressed_sd[final_param_name] = state_dict[param_name]

    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")
    print(f"Number of params before for distillation: {len(state_dict.keys())}")

    print(f"Save transferred checkpoint to {save_checkpoint}")
    torch.save(compressed_sd, save_checkpoint)
