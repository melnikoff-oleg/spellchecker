# https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/scripts/extract.py
import torch
from transformers import BartForConditionalGeneration
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


if __name__ == "__main__":

    init_checkpoint = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent_v0_214056.pt'
    save_checkpoint = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-214056-distil-dec-05.pt'

    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # print(model.config)
    config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
    model = BartForConditionalGeneration(config)
    model.load_state_dict(torch.load(init_checkpoint))

    state_dict = model.state_dict()
    # print(state_dict.keys())
    compressed_sd = {}

    # bad_layers = ['decoder.layers.1.', 'decoder.layers.2.', 'decoder.layers.4.']
    # mapping = {'decoder.layers.3.': 'decoder.layers.1.', 'decoder.layers.5.': 'decoder.layers.2.'}
    # bad_layers = []
    # mapping = {}
    bad_layers = ['decoder.layers.2.']
    mapping = {'decoder.layers.3.': 'decoder.layers.2.', 'decoder.layers.4.': 'decoder.layers.3.',
               'decoder.layers.5.': 'decoder.layers.4.'}

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
            # if param_name in mapping.keys():
            #     compressed_sd[mapping[param_name]] = state_dict[param_name]
            # else:
            compressed_sd[final_param_name] = state_dict[param_name]

    # for param_name in state_dict.keys():
    #     ok = True
    #     for bl in bad_layers:
    #         if bl in param_name:
    #             ok = False
    #     if ok:
    #         compressed_sd[param_name] = state_dict[param_name]


    # prefix = 'model'
    #
    # # Embeddings #
    # for param_name in ["shared.weight"]:
    #     compressed_sd[f"{prefix}.{param_name}"] = state_dict[f"{prefix}.{param_name}"]
    #
    # # Encoder Blocks #
    # for param_name in ["embed_tokens.weight", "embed_positions.weight", "layernorm_embedding.weight",
    #                    "layernorm_embedding.bias"]:
    #     compressed_sd[f"{prefix}.encoder.{param_name}"] = state_dict[f"{prefix}.encoder.{param_name}"]
    # std_idx = 0
    # for teacher_idx in [0, 3, 5]:
    #     for layer in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj",
    #                   "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"]:
    #         for w in ["weight", "bias"]:
    #             compressed_sd[f"{prefix}.encoder.layers.{std_idx}.{layer}.{w}"] = state_dict[
    #                 f"{prefix}.encoder.layers.{teacher_idx}.{layer}.{w}"
    #             ]
    #     std_idx += 1
    #
    # # Decoder Blocks #
    # for param_name in ["embed_tokens.weight", "embed_positions.weight", "layernorm_embedding.weight",
    #                    "layernorm_embedding.bias"]:
    #     compressed_sd[f"{prefix}.decoder.{param_name}"] = state_dict[f"{prefix}.decoder.{param_name}"]
    # std_idx = 0
    # for teacher_idx in [0, 3, 5]:
    #     for layer in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj",
    #                   "self_attn_layer_norm", "encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.q_proj",
    #                   "encoder_attn.out_proj", "encoder_attn_layer_norm", "fc1", "fc2", "final_layer_norm"]:
    #         for w in ["weight", "bias"]:
    #             compressed_sd[f"{prefix}.decoder.layers.{std_idx}.{layer}.{w}"] = state_dict[
    #                 f"{prefix}.decoder.layers.{teacher_idx}.{layer}.{w}"
    #             ]
    #     std_idx += 1
    #
    # # Language Modeling Head ###s
    # compressed_sd["lm_head.weight"] = state_dict["lm_head.weight"]

    # print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")
    print(f"Number of params before for distillation: {len(state_dict.keys())}")

    print(f"Save transferred checkpoint to {save_checkpoint}")
    torch.save(compressed_sd, save_checkpoint)
