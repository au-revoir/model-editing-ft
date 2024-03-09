import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ScriptArguments
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

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

class CustomDPODataCollatorWithPadding(DPODataCollatorWithPadding):
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        consistency_text:str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            if consistency_text != None:
                consistency_tokens = self.tokenizer(consistency_text, add_special_tokens=False, max_length=256, truncation=True)

            chosen_target_ids = [tok for tok in chosen_tokens["input_ids"] if tok != self.tokenizer.eos_token_id]
            rejected_target_ids  = [tok for tok in rejected_tokens["input_ids"] if tok != self.tokenizer.eos_token_id]

            eos_token_id = self.tokenizer.eos_token_id

            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            if consistency_text != None:
                eos_indices_consistency = [i for i, x in enumerate(consistency_tokens["input_ids"]) if x == eos_token_id]

            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            if consistency_text != None:
                new_consistency_attention_mask = [
                    0 if i in eos_indices_consistency else p for i, p in enumerate(consistency_tokens["attention_mask"])
                ]

            prompt_tokens["attention_mask"] = new_attention_mask
            if consistency_text != None:
                consistency_tokens["attention_mask"] = new_consistency_attention_mask

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

            if consistency_text != None:
                for k, toks in {
                    "chosen": chosen_sequence_tokens,
                    "rejected": rejected_sequence_tokens,
                    "prompt": prompt_tokens,
                    "consistency": consistency_tokens

                }.items():
                    for type_key, tokens in toks.items():
                        if type_key == "token_type_ids":
                            continue
                        batch[f"{k}_{type_key}"] = tokens
            else:
                for k, toks in {
                    "chosen": chosen_sequence_tokens,
                    "rejected": rejected_sequence_tokens,
                    "prompt": prompt_tokens

                }.items():
                    for type_key, tokens in toks.items():
                        if type_key == "token_type_ids":
                            continue
                        batch[f"{k}_{type_key}"] = tokens
        
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
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"],

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
        batch["consistency_text"] = consistency_text
        batch["chosen_target_ids"] = chosen_target_ids
        batch["rejected_target_ids"] = rejected_target_ids

        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            if "consistency_text" in feature.keys():
                consistency_text = feature["consistency_text"]
            else:
                consistency_text = None

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, consistency_text)
            tokenized_batch.append(batch_element)

        return self.collate(tokenized_batch)

class ComputeCrossEntropyLoss:
    ignore_index = -100
    def __call__(self, model_output, labels, chosen_targets, rejected_targets, mask_prompt, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        if mask_prompt:
            masks = torch.zeros_like(labels, dtype=torch.bool)
            for i, sublist in enumerate(chosen_targets):
                for value in sublist:
                    masks[i] |= (labels[i] == value)
            labels[~masks] = -100

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        
        padding_mask = labels.eq(self.ignore_index)

        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        
        nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements

        return nll_loss

class CustomTrainer(DPOTrainer):

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
        mask_prompt: Optional[bool] = False,
        include_consistency_text: Optional[bool] = False
        ):
        self.is_encoder_decoder = is_encoder_decoder

        data_collator = CustomDPODataCollatorWithPadding(
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
        self.mask_prompt = mask_prompt
        self.compute_cross_entropy = ComputeCrossEntropyLoss()
        self.include_consistency_text = include_consistency_text

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (   
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs=False,):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs_lang_modeling = {}
        inputs_lang_modeling["input_ids"] = inputs["chosen_input_ids"]
        inputs_lang_modeling["attention_mask"] = inputs["chosen_attention_mask"]

        if self.include_consistency_text:
            consistency_lang_modeling = {}
            consistency_lang_modeling["input_ids"] = inputs["consistency_input_ids"]
            consistency_lang_modeling["attention_mask"] = inputs["consistency_attention_mask"]
            consistency_lang_modeling["labels"] = inputs["consistency_input_ids"]
            outputs_consistency_lang_modeling = model(**consistency_lang_modeling)
            loss_consistency = outputs_consistency_lang_modeling["loss"]

        if args.alpha_loss != 0:
            outputs = model(**inputs_lang_modeling)
            inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()
            loss = self.compute_cross_entropy(outputs, inputs_lang_modeling["labels"], 
                                              inputs["chosen_target_ids"], inputs["rejected_target_ids"], 
                                              self.mask_prompt, shift_labels=True)
        else:
            loss = 0.0

        if args.alpha_loss != 1:
            loss_dpo, metrics = self.get_batch_metrics(model, dpo_inputs, train_eval="train")

            if self.accelerator.is_main_process:
                self.store_metrics(metrics, train_eval="train")
            if return_outputs:
                raise NotImplementedError
        else:
            loss_dpo = 0

        if self.include_consistency_text:
            return (1 - args.consistency_loss_factor) * loss + args.consistency_loss_factor * loss_consistency

        return args.alpha_loss * loss + (1 - args.alpha_loss) * loss_dpo
