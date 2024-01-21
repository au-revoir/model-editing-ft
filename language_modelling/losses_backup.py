import torch
from trl import DPOTrainer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
class CustomDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False,):
        #print("here")
        alpha = 0.6
        #quit()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        print("inputs: ", inputs)
        decoded = tokenizer.decode(inputs["chosen_input_ids"][0])
        print("decoded: ", decoded)
        #quit()
        # Remove occurrences of 50256
        #tensor_without_50256 = torch.masked_select(inputs["chosen_input_ids"], inputs["chosen_input_ids"] != 50256).view(1, -1)
        #attention_mask = (tensor_without_50256 != 50256).int()
        #print("tensor_without_50256: ", tensor_without_50256)
        #print("attention_mask: ", attention_mask)
        #quit()

        inputs_lang_modeling = {}

        input_ids_lm = torch.tensor([[[  464,  2802, 11880,   286, 39808,  7491,  5034,  2821,   318,  3594,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256]]]).to("cuda")

        attention_mask_lm = torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0]]]).to("cuda")

        labels_lm = torch.tensor([[[  464,  2802, 11880,   286, 39808,  7491,  5034,  2821,   318,  3594,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
          50256, 50256, 50256, 50256, 50256, 50256, 50256]]]).to("cuda")

        tensor_without_50256 = torch.masked_select(input_ids_lm, input_ids_lm != 50256).view(1, -1)
        attention_mask = (tensor_without_50256 != 50256).int()






        #inputs_lang_modeling["input_ids"] = inputs["chosen_input_ids"]
        #inputs_lang_modeling["attention_mask"] = inputs["chosen_attention_mask"]
        #inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()#inputs["chosen_labels"]
        
        inputs_lang_modeling["input_ids"] = tensor_without_50256
        inputs_lang_modeling["attention_mask"] = attention_mask#attention_mask
        inputs_lang_modeling["labels"] = tensor_without_50256

        #inputs_lang_modeling["labels"] = inputs["chosen_labels"]
        #inputs_lang_modeling["input_ids"] = torch.tensor()
        print("inputs_lang_modeling: ", inputs_lang_modeling)
        print("tokenizer.decode(torch.tensor([318])):", tokenizer.decode(torch.tensor([318])))
        print("tokenizer.decode(torch.tensor([3594])):", tokenizer.decode(torch.tensor([3594])))

        print("tokenizer.decode(torch.tensor([220])):", tokenizer.decode(torch.tensor([220])))
        print("tokenizer.decode(torch.tensor([15823])):", tokenizer.decode(torch.tensor([15823])))
        #quit()
        outputs = model(**inputs_lang_modeling)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        print("LM loss: ", loss)
        #quit()
        loss_dpo, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        #print("DPO loss: ", loss_dpo)
        #print("here")
        quit()
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            raise NotImplementedError
        return loss#loss_dpo
