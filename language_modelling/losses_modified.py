import torch
import torch.nn as nn
from datasets import Dataset
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from trl.trainer import DPOTrainer
from transformers import HfArgumentParser
from transformers.trainer_utils import has_length
from transformers.utils import logging
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, TrainOutput
from trl.trainer.utils import DPODataCollatorWithPadding

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from config_dpo import ScriptArguments
#from globals import eval_method
import globals


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logger = logging.get_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

class CustomDPODataCollatorWithPadding(DPODataCollatorWithPadding):
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        lm_prompt:str
    ) -> Dict:
        #print("prompt: ", prompt)
        #print("lm_prompt: ", lm_prompt)
        #print("batch: ", batch)
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}


        #print("prompt: ", prompt)
        if not self.is_encoder_decoder:
            #print("inside not is_encoder_decoder")
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            lm_tokens = self.tokenizer(lm_prompt, add_special_tokens=False)

            #print("lm_tokens: ", lm_tokens)
            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            eos_indices_lm = [i for i, x in enumerate(lm_tokens["input_ids"]) if x == eos_token_id]
            #print("eos_indices_lm: ", eos_indices_lm)
            #quit()
            

            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]

            new_lm_attention_mask = [
                0 if i in eos_indices_lm else p for i, p in enumerate(lm_tokens["attention_mask"])
            ]

            #print("new_lm_attention_mask: ", new_lm_attention_mask)
            #quit()
            prompt_tokens["attention_mask"] = new_attention_mask
            lm_tokens["attention_mask"] = new_lm_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
                "lm": lm_tokens
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens
            #print("batch: ", batch)

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected
        batch["lm"] = lm_prompt 
        #quit()
        return batch
        #quit()
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        #print("features: ", features)
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            lm_prompt = feature["lm_prompt"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, lm_prompt)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


class CustomDPOTrainer(DPOTrainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        eval_edit_data: Optional[List] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        ):
        self.eval_edit_data = eval_edit_data
        self.is_encoder_decoder = is_encoder_decoder

        data_collator = CustomDPODataCollatorWithPadding(
        #data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
            )

        super().__init__(model, ref_model, beta, loss_type, args, 
                         data_collator, label_pad_token_id, padding_value, 
                         truncation_mode, train_dataset, eval_dataset, 
                         tokenizer, model_init, callbacks, optimizers,
                         preprocess_logits_for_metrics, max_length,
                         max_prompt_length, max_target_length, peft_config,
                         is_encoder_decoder, disable_dropout, generate_during_eval,
                         compute_metrics, model_init_kwargs, ref_model_init_kwargs)
        self.step_idx_for_eval = 0


    def compute_loss(self, model, inputs, return_outputs=False,):
        #print("here")
        #script_args.alpha = 0.7
        #quit()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs_lang_modeling = {}


        #print("inputs: ", inputs)
        #quit()

        #print("inputs['chosen_input_ids']: ", inputs["chosen_input_ids"])
        #print("inputs['chosen_attention_mask']: ", inputs["chosen_attention_mask"])
        #print("inputs['chosen_input_ids'].shape: ", inputs["chosen_input_ids"].shape)
        #print("inputs['lm_input_ids']: ", inputs["lm_input_ids"].shape)
        #print("inputs['lm_attention_mask']: ", inputs["lm_attention_mask"])

        #quit()
        
        #"""
        #inputs_lang_modeling["input_ids"] = inputs["lm_input_ids"]
        #inputs_lang_modeling["attention_mask"] = inputs["lm_attention_mask"]
        #inputs_lang_modeling["labels"] = inputs["lm_input_ids"].clone()#inputs["chosen_labels"]
        #print("inputs_lang_modeling['input_ids']: ", inputs_lang_modeling["input_ids"].shape)
        #print("inputs_lang_modeling['attention_mask']: ", inputs_lang_modeling["attention_mask"].shape)
       
        #print('inputs["lm_attention_mask"].sum(dim=0) > 0: ', inputs["lm_attention_mask"].sum(dim=1) > 0)
        #inputs_lang_modeling["input_ids"] = inputs["lm_input_ids"][inputs["lm_attention_mask"].sum(dim=1) > 0]
        #inputs_lang_modeling["attention_mask"] = inputs["lm_attention_mask"][inputs["lm_attention_mask"].sum(dim=1) > 0]
        #inputs_lang_modeling["labels"] = inputs["lm_input_ids"][inputs["lm_attention_mask"].sum(dim=1) > 0]
        #"""
        inputs_lang_modeling["input_ids"] = inputs["chosen_input_ids"]
        inputs_lang_modeling["attention_mask"] = inputs["chosen_attention_mask"]
        inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()#inputs["chosen_labels"]
        #"""
        #print("inputs_lang_modeling: ", inputs_lang_modeling)
        #print('inputs_lang_modeling["input_ids"].nelement(): ', inputs_lang_modeling["input_ids"].nelement())
        #quit()
        #if False:
        if script_args.alpha_loss != 0:
            #if False:
            #print("alpha is non zero")
            if inputs["lm"] ==  ['<|endoftext|>']: #or inputs_lang_modeling["input_ids"].nelement() == 0:
                #print("skip_lm")
                loss = 0.0
            else:
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
        else:
            loss = 0.0

        if script_args.alpha_loss != 1:# or inputs_lang_modeling["input_ids"].shape[0] != 0: 
            loss_dpo, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
             
            #print("DPO loss: ", loss_dpo)
            #print("here")
            #quit()
            # force log the metrics
            if self.accelerator.is_main_process:
                self.store_metrics(metrics, train_eval="train")
            if return_outputs:
                raise NotImplementedError
        else:
            loss_dpo = 0

        #if inputs["chosen_input_ids"] ==  ['<|endoftext|>']:
        #    loss_dpo = 0

        return script_args.alpha_loss * loss + (1 - script_args.alpha_loss) * loss_dpo
        #return loss + loss_dpo


    """

    def get_parameter(self, model, name):
        #Finds the named parameter within the given model.
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    def reset_model(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                w = self.get_parameter(self.ref_model, name)
                for n, p in self.model.named_parameters():
                    if name == n:
                        p[...] = w.detach().clone()#.to("cuda")
        print("Model reset")

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        print("len(batch_size): ", batch_size)
        print("self.step_idx_for_eval: ", self.step_idx_for_eval)
        #quit()
        if batch_size != 1:
            raise NotImplementedError(f"batch_size={batch_size} not supported yet because of evaluation")
        self.state_global_step, train_loss, metrics = super()._inner_training_loop(batch_size, args, resume_from_checkpoint, ignore_keys_for_eval)
        globals.eval_method.run_eval(self.model, self.step_idx_for_eval)
        self.step_idx_for_eval += batch_size
        print("self.step_idx_for_eval: ", self.step_idx_for_eval)
        #quit()

        self.reset_model()
        return TrainOutput(self.state.global_step, train_loss, metrics)
   """
