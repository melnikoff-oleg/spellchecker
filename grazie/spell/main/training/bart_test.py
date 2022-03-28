from transformers import BartTokenizer, BartForConditionalGeneration
import torch

if __name__ == '__main__':

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    TXT = "My friends are <mask> but they eat too many carbs."
    TXT = "are My friends great but they eat too many carbs."

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    # input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    # logits = model(input_ids).logits
    device = torch.device('cuda')
    model = model.to(device)

    result_ids = model.generate(tokenizer([TXT], return_tensors='pt').to(device)["input_ids"],
                                num_beams=5, min_length=5, max_length=100)
    print('Quality check:')
    print('Query noise:', TXT)
    print('Query gt:', 'Lol')
    print('Answer:', tokenizer.batch_decode(result_ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False))

    # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # probs = logits[0, masked_index].softmax(dim=0)
    # values, predictions = probs.topk(5)

    # print(tokenizer.decode(predictions).split())

