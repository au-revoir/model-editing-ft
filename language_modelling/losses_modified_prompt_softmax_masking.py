import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

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
        #tag:str
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

            #print("chosen_tokens: ", chosen_tokens)
            #print("rejected_tokens: ", rejected_tokens)
            #print("prompt_tokens: ", prompt_tokens)

            chosen_target_ids = [tok for tok in chosen_tokens["input_ids"] if tok != self.tokenizer.eos_token_id]
            rejected_target_ids  = [tok for tok in rejected_tokens["input_ids"] if tok != self.tokenizer.eos_token_id]
            
            #quit()

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
            #print("chosen_tokens: ", chosen_tokens)
            #quit()
            #print("prompt_tokens['input_ids']: ", prompt_tokens["input_ids"])
            end_position_prompt = len(prompt_tokens['input_ids'])
            #quit()
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

            #print("chosen_sequence_tokens: ", chosen_sequence_tokens)
            #print("rejected_sequence_tokens: ", rejected_sequence_tokens)
            #quit()
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
        batch["lm"] = lm_prompt 
        #batch["tag"] = tag
        batch["target_position"] = end_position_prompt
        batch["chosen_target_ids"] = chosen_target_ids
        batch["rejected_target_ids"] = rejected_target_ids
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
            #tag = feature["tag"]

            #print("prompt: ", prompt)
            #print("chosen: ", chosen)
            #quit()

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, lm_prompt)#, tag)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


#@dataclass
class SelectiveCrossEntropyLoss:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    #epsilon: float = 0.1
    epsilon: float = 0.0
    ignore_index: int = -100
    """
    def __init__(self, ignore_template_labels, mask_softmax, no_masking_softmax):
        self.ignore_template_labels  = ignore_template_labels
        self.mask_softmax = mask_softmax
        self.no_masking_softmax = no_masking_softmax
   """
    def __call__(self, model_output, labels, position, chosen_targets, rejected_targets, ignore_template_labels, mask_softmax, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        #print("position: ", position)
        #print("chosen_targets: ", chosen_targets)
        #print("rejected_targets: ", rejected_targets)
        #quit()
        #print("before shift logits: ", logits)
        #print("before shift logits.shape: ", logits.shape)
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        #print("before labels: ", labels)
        #quit()
        
        #logits_copy = logits.clone()

        #print("after shift logits:  ", logits)
        batch_indices = [i for i in range(logits.shape[0])]

        
        #quit()
        if ignore_template_labels:
        #if cross_entropy_type == "ignore_template_labels":
            position_before = [x - 1 for x in position] #we shift labels to the right by 1 so subtract by 1
            label_at_position = labels[batch_indices, position_before]
            #print("position_before: ", position_before)
            #position_next = [x+1 for x in position]
            #positions_after = [list(range(x, labels.shape[1])) for x in position_next]
            #labels_after_position = labels[batch_indices, positions_after]
            labels[batch_indices, :] = -100
            labels[batch_indices, position_before] = label_at_position
            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            #labels[batch_indices, positions_after] = labels_after_position
        #print("after -100 labels: ", labels)
        #quit()
        if mask_softmax:
            #print("inside mask_softmax")
        #if cross_entropy_type == "mask_softmax":
            #position_after = [x + 1 for x in position]
            chosen_target_token_ids_flattened = [x for xs in chosen_targets for x in xs]
            rejected_target_token_ids_flattened = [x for xs in rejected_targets for x in xs]


            #position = [x-1 for x in position]

            #print("position - 1: ", position)

            #quit()
            logits_chosen_targets = logits[batch_indices, position, chosen_target_token_ids_flattened]
            logits_rejected_targets = logits[batch_indices, position, rejected_target_token_ids_flattened]

            #print("logits_chosen_targets: ", logits_chosen_targets)

            #Set all the logits of in the position to -inf first and then replace the chosen_targets_logits to their correct position
            logits[batch_indices, position, :] = float("-inf")
            #print("before logits[batch_indices, position, :]: ", logits[batch_indices, position, :])#chosen_target_token_ids_flattened])
            logits[batch_indices, position, chosen_target_token_ids_flattened] = logits_chosen_targets
            #print("after logits[batch_indices, position, :]: ", logits[batch_indices, position, :])#chosen_target_token_ids_flattened])
            logits[batch_indices, position, rejected_target_token_ids_flattened] = logits_rejected_targets
            #print("after shift labels: ", labels)
            #print("logits: ", logits)
            #quit()

        #print("after shift logits.shape:  ", logits.shape)
        #print("after shift labels.shape: ", labels.shape)
        #quit()
        #logits[[0,1], :] = torch.where[0,:] 

            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            log_probs_chosen_targets = log_probs[batch_indices, position, chosen_target_token_ids_flattened]
            log_probs_rejected_targets = log_probs[batch_indices, position, rejected_target_token_ids_flattened]

            log_probs[batch_indices, position, :] = 0.0
            log_probs[batch_indices, position, chosen_target_token_ids_flattened] = log_probs_chosen_targets
            log_probs[batch_indices, position, rejected_target_token_ids_flattened] = log_probs_rejected_targets

        else: #self.no_masking_softmax:
        #elif cross_entropy_type == "no_masking_softmax":
            log_probs = -nn.functional.log_softmax(logits, dim=-1)
        #print("log_probs: ", log_probs)
        #quit()
        #print("log_probs[batch_indices, position, chosen_target_token_ids_flattened]: ", log_probs[batch_indices, position, chosen_target_token_ids_flattened])
        #print("log_probs[batch_indices, position, rejected_target_token_ids_flattened]: ", log_probs[batch_indices, position, rejected_target_token_ids_flattened])
        #print("log_probs.shape: ", log_probs.shape)
        #quit()
        if labels.dim() == log_probs.dim() - 1:
            #print("inside labels.dim() == log_probs.dim() - 1")
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        #print("padding_mask: ", padding_mask)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        #print("labels after clamp: ", labels)
        #quit()
        nll_loss = log_probs.gather(dim=-1, index=labels)
        #print("nll_loss.shape: ", nll_loss.shape)
        #quit()
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        
        #print("padding_mask.numel(): ", padding_mask.numel())
        #print("padding_mask.long().sum(): ", padding_mask.long().sum())
        #quit()
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        #print("loss: ", nll_loss)
        #quit()
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


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
        #cross_entropy_type: str = "no_masking",
        ignore_template_labels: Optional[bool] = False,
        mask_softmax: Optional[bool] = False
        #no_masking_softmax: Optional[bool] = False
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
        #print("self.label_smoother: ", self.label_smoother)
        #self.cross_entropy_type = cross_entropy_type
        self.ignore_template_labels = ignore_template_labels
        self.mask_softmax = mask_softmax
        self.selective_cross_entropy = SelectiveCrossEntropyLoss()
        #quit()

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        #tags,
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
        #tags = torch.tensor(tags)
        
        #print("tags: ", tags)
        #quit()

        if reference_free:
            ref_logratios = 0

        #beta = torch.tensor([self.beta]).to(pi_logratios.get_device())
        #beta = beta.repeat(policy_chosen_logps.shape[0], )
        #beta = torch.where(tags==False, 0.1, self.beta).to(pi_logratios.get_device())
        #beta = torch.where(tags==False, self.beta, self.beta).to(pi_logratios.get_device())
        #beta = beta[tags=="paraphrase_tag"] = torch.tensor(0.1)
        #print("beta before: ", beta)
        #beta = beta.permute(*torch.arange(beta.ndim - 1, -1, -1))
        #beta = beta.T
        #print("beta after: ", beta)

        logits = pi_logratios - ref_logratios

        #print("logits: ", logits)
        #print("logits.shape: ", logits.shape)

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
            #losses = -F.logsigmoid(torch.mul(beta, logits))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
            #losses = torch.relu(1 - beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")
        #print("losses: ", losses)
        #print("losses.shape: ", losses.shape)
    
        #quit()
        ##print("policy_chosen_logps: ", policy_chosen_logps)
        #print("reference_chosen_logps: ", reference_chosen_logps)
        #print("policy_chosen_logps.shape: ", policy_chosen_logps.shape)
        #print("reference_chosen_logps.shape: ", reference_chosen_logps.shape)
        #quit()
        #print("policy_chosen_logps - reference_chosen_logps: ", policy_chosen_logps - reference_chosen_logps)
        #print("policy_rejected_logps - reference_rejected_logps: ", policy_rejected_logps - reference_rejected_logps)
        #beta = torch.tensor([self.beta])
        #beta = beta.repeat(policy_chosen_logps.shape[0], )
        #print("beta: ", beta)
        #print("beta.shape: ", beta.shape)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        #chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        #rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        #print("chosen_rewards: ", chosen_rewards)
        #print("rejected_rewards: ", rejected_rewards)
        #quit()


        return losses, chosen_rewards, rejected_rewards

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        #print("batch: ", batch)
        #quit()
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

        #print("policy_chosen_logps.shape: ", policy_chosen_logps.shape)
        #print("policy_rejected_logps.shape: ", policy_rejected_logps.shape)
        #print("reference_chosen_logps.shape: ", reference_chosen_logps.shape)
        #print("reference_rejected_logps: ", reference_rejected_logps.shape)
        #quit()
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
            #batch["tag"]
        )
        #quit()
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

    def lm_loss(self, model, inputs):
        
        print("inputs['input_ids']: ", inputs["input_ids"])
        print("decoded input_ids: ", tokenizer.batch_decode(inputs["input_ids"]))
        outputs = model(**inputs)
        print("outputs['logits']: ", outputs["logits"])
        print("outputs['logits'].shape: ", outputs["logits"].shape)
        quit()

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                print("inside MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values")
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
        quit()


    def compute_loss(self, model, inputs, return_outputs=False,):
        #print("here")
        #script_args.alpha = 0.7
        #quit()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        #print("labels: ", labels)
        #quit()

        inputs_lang_modeling = {}

        #print("inputs['chosen_target_ids']: ", inputs["chosen_target_ids"])
        for i in inputs["chosen_target_ids"]:
            if len(i) > 1:
                print("inputs")
                quit()

        #print("input$$s: ", inputs)
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
        #inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()#inputs["chosen_labels"]
        #"""
        #print("inputs_lang_modeling: ", inputs_lang_modeling)
        #print('inputs_lang_modeling["input_ids"]: ', inputs_lang_modeling["input_ids"].shape)
        #print('inputs_lang_modeling["input_ids"].nelement(): ', inputs_lang_modeling["input_ids"].nelement())
        #print("inputs['chosen_target_ids']: ", inputs["chosen_target_ids"])
        #print("inputs['rejected_target_ids']: ", inputs["rejected_target_ids"])
        #quit()
        #if False:
        if script_args.alpha_loss != 0:
            #if False:
            #print("alpha is non zero")
            if inputs["lm"] ==  ['<|endoftext|>']: #or inputs_lang_modeling["input_ids"].nelement() == 0:
                #print("skip_lm")
                loss = 0.0
            else:
                #print("inside else here")
                #loss_model = self.lm_loss(model, inputs_lang_modeling)
                outputs = model(**inputs_lang_modeling)
                #loss = outputs["loss"]
                inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()
                loss = self.selective_cross_entropy(outputs, inputs_lang_modeling["labels"], inputs["target_position"], inputs["chosen_target_ids"], inputs["rejected_target_ids"], self.ignore_template_labels, self.mask_softmax, shift_labels=True)
                #print("outputs['loss']: ", outputs["loss"])
                #print("loss_selected: ", loss_selected)

                #quit()
                """
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
                """
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
