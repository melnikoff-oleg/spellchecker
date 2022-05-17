from model.spellcheck_model import *
from training.common_parts import *
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    # data and model prep
    config = {
        "_name_or_path": "facebook/bart-base",
        "activation_dropout": 0.1,
        "activation_function": "gelu",
        "add_bias_logits": False,
        "add_final_layer_norm": False,
        "architectures": [
            "BartModel"
        ],
        "attention_dropout": 0.1,
        "bos_token_id": 0,
        "classif_dropout": 0.1,
        "classifier_dropout": 0.0,
        "d_model": 768,
        "decoder_attention_heads": 12,
        "decoder_ffn_dim": 3072,
        "decoder_layerdrop": 0.0,
        "decoder_layers": 3,
        "decoder_start_token_id": 2,
        "dropout": 0.1,
        "early_stopping": True,
        "encoder_attention_heads": 12,
        "encoder_ffn_dim": 3072,
        "encoder_layerdrop": 0.0,
        "encoder_layers": 3,
        "eos_token_id": 2,
        "forced_bos_token_id": 0,
        "forced_eos_token_id": 2,
        "gradient_checkpointing": False,

        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
        },
        "init_std": 0.02,
        "is_encoder_decoder": True,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
        },
        "max_position_embeddings": 1024,
        "model_type": "bart",
        "no_repeat_ngram_size": 3,
        "normalize_before": False,
        "normalize_embedding": True,
        "num_beams": 4,
        "num_hidden_layers": 6,
        "pad_token_id": 1,
        "scale_embedding": False,
        "task_specific_params": {
            "summarization": {
                "length_penalty": 1.0,
                "max_length": 128,
                "min_length": 12,
                "num_beams": 4
            },
            "summarization_cnn": {
                "length_penalty": 2.0,
                "max_length": 142,
                "min_length": 56,
                "num_beams": 4
            },
            "summarization_xsum": {
                "length_penalty": 1.0,
                "max_length": 62,
                "min_length": 11,
                "num_beams": 6
            }
        },
        "torch_dtype": "float32",
        "transformers_version": "4.18.0",
        "use_cache": True,
        "vocab_size": 50265
    }
    checkpoint = PATH_PREFIX + 'training/checkpoints/oldbart-en05dec05.pt'
    train_data, val_data = get_oldbart_training_dataset()
    tokenizer, model = bart_model_init(checkpoint, BartConfig(**config))

    # set important learning params ------------------------------------------
    device_name = 'cuda:2'
    model_version = 0
    model_name = 'oldbart-en05dec05'
    lr = 0.0001
    test_mode = False
    st_epoch = 0
    batch_size = 32
    num_epochs = 2
    print_n_batches = 2000
    spellcheck_class = OldBartChecker
    save_model_interval = 3876 # 4097007 # 3969782

    # setup all remaining parts for learning
    launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint)


if __name__ == '__main__':
    main()
