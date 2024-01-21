import torch
import torch.nn as nn
from datasets import Dataset
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from trl import DPOTrainer
from transformers import HfArgumentParser
from transformers.trainer_utils import has_length
from transformers.utils import logging
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

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


        #print("inputs['chosen_input_ids'].shape: ", inputs["chosen_input_ids"].shape)
        #quit()

        inputs_lang_modeling["input_ids"] = inputs["chosen_input_ids"]
        inputs_lang_modeling["attention_mask"] = inputs["chosen_attention_mask"]
        inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()#inputs["chosen_labels"]
        #print("inputs_lang_modeling: ", inputs_lang_modeling)
        
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

        loss_dpo, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        #print("DPO loss: ", loss_dpo)
        #print("here")
        #quit()
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            raise NotImplementedError
        return script_args.alpha_loss * loss + (1 - script_args.alpha_loss) * loss_dpo

    """
    def reset_model(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                w = get_parameter(model_ref, name)
                for n, p in self.model.named_parameters():
                    if name == n:
                        p[...] = w.detach().clone()#.to("cuda")


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        print("len(batch_size): ", batch_size)
        print("self.step_idx_for_eval: ", self.step_idx_for_eval)
        if batch_size != 1:
            raise NotImplementedError(f"batch_size={batch_size} not supported yet because of evaluation")
        self.state_global_step, train_loss, metrics = super()._inner_training_loop(batch_size, args, resume_from_checkpoint, ignore_keys_for_eval)
        globals.eval_method.run_eval(self.model, self.step_idx_for_eval)
        self.step_idx_for_eval += batch_size
        #quit()
        self.reset_model()
        return TrainOutput(self.state.global_step, train_loss, metrics)
    """
