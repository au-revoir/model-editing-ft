import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label_pad_token_id = -100
        self.padding_value = 0

    def collate(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch


    def tokenizer_batch_element(self, prompt, chosen):
        batch = {}
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

        chosen_target_ids = [tok for tok in chosen_tokens["input_ids"] if tok != self.tokenizer.eos_token_id]

        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == self.tokenizer.eos_token_id]
        new_attention_mask = [0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])]
        prompt_tokens["attention_mask"] = new_attention_mask

        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == self.tokenizer.eos_token_id]
        new_attention_mask_chosen = [0 if i in eos_indices_prompt else p for i, p in enumerate(chosen_tokens["attention_mask"])]
        chosen_tokens["attention_mask"] = new_attention_mask_chosen

        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(prompt_tokens["input_ids"])

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "prompt": prompt_tokens
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["chosen_target_ids"] = chosen_target_ids
        return batch
        
    def apply_tokenization(self, features, epoch_num):
        tokenizer_batch = []
        random.seed(epoch_num)
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]

            batch_element = self.tokenizer_batch_element(prompt, chosen)
            tokenizer_batch.append(batch_element)
        random.shuffle(tokenizer_batch)
        return self.collate(tokenizer_batch)

class CustomCrossEntropyLoss:
    ignore_index = -100

    def __call__(self, model_output, labels, chosen_targets, ignore_template_labels, shift_labels=True):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        batch_indices = [i for i in range(logits.shape[0])]

        if ignore_template_labels:
            masks = torch.zeros_like(labels, dtype=torch.bool)
            for i, sublist in enumerate(chosen_targets):
                for value in sublist:
                    masks[i] |= (labels[i] == value)
            labels[~masks] = -100

        log_probs = -nn.functional.log_softmax(logits, dim=-1)

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)

        labels = torch.clamp(labels, min=0)

        nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements

        return nll_loss
        
class Trainer:
    def __init__(self):
        self.custom_ce_loss = CustomCrossEntropyLoss()
        self.counter = 0
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

    def compute_loss(self, model, inputs, ignore_template_labels):
        lm_inputs = {}
        lm_inputs["input_ids"] = inputs["chosen_input_ids"].to(model.device)
        lm_inputs["attention_mask"] = inputs["chosen_attention_mask"].to(model.device)

        outputs = model(**lm_inputs)#.to(model.device()))
        loss = self.custom_ce_loss(outputs, inputs["chosen_input_ids"].to(model.device), inputs["chosen_target_ids"], ignore_template_labels, shift_labels=True)
        self.counter += 1
        return loss
