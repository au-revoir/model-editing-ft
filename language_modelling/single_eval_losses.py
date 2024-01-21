import math
import torch
import torch.nn as nn
import time
from datasets import Dataset
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, RandomSampler

from packaging import version
from trl import DPOTrainer
from transformers import HfArgumentParser
from transformers.trainer_utils import has_length, speed_metrics
from transformers.utils import logging, is_sagemaker_mp_enabled, is_datasets_available, is_torch_tpu_available
from transformers.trainer_callback import TrainerState
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, TrainOutput, seed_worker, find_executable_batch_size, set_seed
from trl.trainer.utils import DPODataCollatorWithPadding
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_model_param_count

from accelerate import __version__ as accelerate_version

from config_dpo import ScriptArguments
#from globals import eval_method
import globals

#from dpo_gpt2_with_neighbor_prompts import compare_models

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logger = logging.get_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

if is_datasets_available():
    import datasets



def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                #print('Mismtach found at', key_item_1[0])
                pass
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
    else:
        print("Models don't match")

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

        if not self.is_encoder_decoder:
            #print("inside not is_encoder_decoder")
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            lm_tokens = self.tokenizer(lm_prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            eos_indices_lm = [i for i, x in enumerate(lm_tokens["input_ids"]) if x == eos_token_id]

            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]

            new_lm_attention_mask = [
                0 if i in eos_indices_lm else p for i, p in enumerate(lm_tokens["attention_mask"])
            ]

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
        state_dict: Optional[Dict] = None,
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
        self.reference_state_dict = state_dict
        self.is_encoder_decoder = is_encoder_decoder
        #print("self.optimizer: ", self.optimizer)

        #quit()
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
        #print("self.args: ", self.args)
        #print("optimizers: ", optimizers)
        #print("self.optimizer: ", self.optimizer)
        #print("self.lr_scheduler: ", self.lr_scheduler)
        #quit()


    def compute_loss(self, model, inputs, return_outputs=False,):
        #print("here")
        #script_args.alpha = 0.7
        #quit()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs_lang_modeling = {}


        #print("\ninputs: ", inputs)


        #print("inputs['chosen_input_ids'].shape: ", inputs["chosen_input_ids"].shape)
        #quit()
        
        inputs_lang_modeling["input_ids"] = inputs["lm_input_ids"].clone()
        inputs_lang_modeling["attention_mask"] = inputs["lm_attention_mask"].clone()
        inputs_lang_modeling["labels"] = inputs["lm_input_ids"].clone()#inputs["chosen_labels"]
        """
        inputs_lang_modeling["input_ids"] = inputs["chosen_input_ids"]
        inputs_lang_modeling["attention_mask"] = inputs["chosen_attention_mask"]
        inputs_lang_modeling["labels"] = inputs["chosen_input_ids"].clone()#inputs["chosen_labels"]
        """
        #print("inputs_lang_modeling: ", inputs_lang_modeling)
        #if False:
        if script_args.alpha_loss != 0:
            #if False:
            #print("alpha is non zero")
            if inputs["lm"] ==  ['<|endoftext|>']:
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

        loss_dpo, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        #print("DPO loss: ", loss_dpo)
        #print("here")
        #quit()
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            raise NotImplementedError
        print("loss_dpo: ", loss_dpo.item())
        return script_args.alpha_loss * loss + (1 - script_args.alpha_loss) * loss_dpo


    

    #def get_parameter(self, model, name):
        """
        #Finds the named parameter within the given model.
        """
    #    for n, p in model.named_parameters():
    #        if n == name:
    #            return p
    #    raise LookupError(name)

    """
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
    def _get_train_sampler_NOTUSE(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        #print("inside _get_train_sampler")
        if self.args.group_by_length:
            #print("inside if self.args.group_by_length")
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                #print("inside if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset)")
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            #print("lengths: ", lengths)
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)


    def get_train_dataloader(self) -> DataLoader:
        #"""
        #Returns the training [`~torch.utils.data.DataLoader`].

        #Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        #training if necessary) otherwise.

        #Subclass and override this method if you want to inject some custom behavior.
        #"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        #current_row = self.train_dataset[self.step_idx_for_eval]
        #current_row_item =  current_row[self.current_item_idx]
        #print("\nself.train_dataset: ", self.train_dataset)
        train_dataset = self.train_dataset[self.step_idx_for_eval]
        #print("before train_dataset: ", train_dataset[:])
        #quit()
        #"""
        #train_dataset = [{"prompt": train_dataset["prompt"][i], "chosen": train_dataset["chosen"][i], 
        #                  "rejected": train_dataset["rejected"][i], "lm_prompt": train_dataset["lm_prompt"][i]} for i in range(len(train_dataset["prompt"]))]
        #train_dataset = Dataset.from_list(train_dataset)
        train_dataset = Dataset.from_dict(train_dataset)
        #"""
        #print("current_row: ", current_row)
        #print("\ncurrent_row_item: ", current_row_item)
        #print("\nafter train_dataset: ", train_dataset[:])

        #print("type: ", type(train_dataset))
        #quit()
        data_collator = self.data_collator
        #print("data_collator: ", data_collator)
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            #print("inside iterableDataset")
            dataloader_params["sampler"] = None#RandomSampler(train_dataset)#self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        #print("dataloader_params: ", dataloader_params)
        #print("\ntrain_dataset right before prepare: ", train_dataset[:])
        #print(train_dataset.__dict__)
        #quit()
        #print("train_dataset[4]: ", train_dataset[4])
        train_loader = DataLoader(train_dataset, **dataloader_params)
        #print("len(train_loader) inside get_train_loader: ", len(train_loader))
        #quit()
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))



    def train(
        self,
        #example_idx: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        #print("\nInside def train")
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        #enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        #self.model = self.call_model_init(trial)
        self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        #print("self._train_batch_size: ", self._train_batch_size)
        #print("args.auto_find_batch_size: ", args.auto_find_batch_size)

        #print("trial: ", trial)
        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        #print("inner_training_loop: ", inner_training_loop.__dict__)
        #quit()
        #quit()
        """
        for example_idx, _ in enumerate(self.train_dataset):
            self.step_idx_for_eval = example_idx
            _ = inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        """    
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
        
        #return _

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
            
            #print("\nEXECUTING _inner_training_loop")
        for example_idx, _ in enumerate(self.train_dataset):
            #self.optimizer, self.lr_scheduler = None, None
            self.optimizer = None
        #if True:
            self.step_idx_for_eval = example_idx
            self.accelerator.free_memory()
            self._train_batch_size = batch_size
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
            # Data loader and number of training steps
            self.current_item_idx = 0
            #print("self.dataset: ", self.dataset)
            train_dataloader = self.get_train_dataloader()

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            #print("\n", train_dataloader.__dict__)
            #for i in train_dataloader:
            #    print(i)
            #print("\nlen(train_dataloader): ", len(train_dataloader))
            #quit()
            len_dataloader = None
            if has_length(train_dataloader):
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_examples = self.num_examples(train_dataloader)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
            elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_examples = total_train_batch_size * args.max_steps
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {args.max_steps}"
                )

            # Compute absolute values for logging, eval, and save if given as ratio
            if args.logging_steps and args.logging_steps < 1:
                args.logging_steps = math.ceil(max_steps * args.logging_steps)
            if args.eval_steps and args.eval_steps < 1:
                args.eval_steps = math.ceil(max_steps * args.eval_steps)
            if args.save_steps and args.save_steps < 1:
                args.save_steps = math.ceil(max_steps * args.save_steps)

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                        " (torch.distributed.launch)."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
            )
            #if True:
            #for example_idx, _ in enumerate(self.train_dataset):
            #model.load_state_dict(self.reference_state_dict)
            self.model.load_state_dict(self.reference_state_dict)
            self.model_wrapped = self.model
            #total_batched_samples = 0
            #self.step_idx_for_eval = example_idx
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            #model.zero_grad()
            #print("example_idx: ", example_idx)
            train_dataloader = self.get_train_dataloader()
            # We need to reset the scheduler, as its parameters may be different on subsequent calls
            if self._created_lr_scheduler:
                self.lr_scheduler = None
                self._created_lr_scheduler = False

            if self.is_deepspeed_enabled:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None

            # Activate gradient checkpointing if needed
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            model = self._wrap_model(self.model_wrapped)

            model.load_state_dict(self.reference_state_dict)


            if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
                self._load_from_checkpoint(resume_from_checkpoint, model)

            # as the model is wrapped, don't use `accelerator.prepare`
            # this is for unhandled cases such as
            # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # prepare using `accelerator` prepare
            #print("use_accelerator_prepare: ", use_accelerator_prepare)
            if use_accelerator_prepare:
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    #print("inside hasattr(self.lr_scheduler, 'step')")
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                #print("if model is not self.model")
                self.model_wrapped = model
                #quit()

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            # deepspeed ckpt loading
            if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            if self.hp_name is not None and self._trial is not None:
                # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
                # parameter to Train when using DDP.
                self.state.trial_name = self.hp_name(self._trial)
            if trial is not None:
                assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
                self.state.trial_params = hp_params(assignments)
            else:
                self.state.trial_params = None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()

            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    for _ in train_dataloader:
                        break

            total_batched_samples = 0
            #self.current_item_idx = 0
                
            #self.step_idx_for_eval += batch_size
            #print("self.train_dataset: ", self.train_dataset)
            #print("len(self.train_dataset): ", len(self.train_dataset))
            #quit()
            #for example_idx, _ in enumerate(self.train_dataset):
            #total_batched_samples = 0
            #self.step_idx_for_eval = example_idx
            #self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            #model.zero_grad()
            #print("example_idx: ", example_idx)
            #train_dataloader = self.get_train_dataloader()
            for epoch in range(epochs_trained, num_train_epochs):
                #train_dataloader = self.get_train_dataloader()
                epoch_iterator = train_dataloader
                print("epoch: ", epoch)
                #print("epoch_iterator: ", epoch_iterator)
                #print("len(epoch_iterator): ", len(epoch_iterator))

                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                self.current_item_idx = 0
                for step, inputs in enumerate(epoch_iterator):
                    #print("step: ", step)
                    #print("\ninputs: ", inputs["prompt"])
                    total_batched_samples += 1
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    #print("\nsteps_trained_in_current_epoch: ", steps_trained_in_current_epoch)
                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc or (
                            version.parse(accelerate_version) <= version.parse("0.20.3")
                        ):
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                                self.optimizer.step()
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()
                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                        print("optimizer_was_run: ", optimizer_was_run)
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                print("inside not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)")
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        """
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                """
                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                #if self.control.should_training_stop:
                #    break
                self.current_item_idx += 1

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sur the model has been saved by process 0.
                if is_torch_tpu_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                    dist.barrier()
                elif is_sagemaker_mp_enabled():
                    smp.barrier()

                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            train_loss = self._total_loss_scalar / self.state.global_step

            metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if checkpoint != self.state.best_model_checkpoint:
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint)

            #self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            #compare_models(model, self.ref_model)
            globals.eval_method.run_eval(model, self.step_idx_for_eval)
            #self.reset_model()
            
            """
            with torch.no_grad():
                for name, param in model.named_parameters():
                    w = self.get_parameter(self.ref_model, name)
                    for n, p in model.named_parameters():
                        if name == n:
                            p[...] = w.detach().clone()#.to("cuda")
            """
            model.load_state_dict(self.reference_state_dict)
            self.model.load_state_dict(self.reference_state_dict)
            self.ref_model.load_state_dict(self.reference_state_dict)
            #print("Model reset")
            #compare_models(model, self.ref_model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def get_parameter(self, ref_model, name):
        #Finds the named parameter within the given model.
        for n, p in ref_model.named_parameters():
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

    """
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):


        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        print("self.train_dataset: ", self.train_dataset)
        #quit()
        train_dataloader = self.get_train_dataloader()
        print(train_dataloader)
        print("args: ", args)
        
        print("self.args: ", self.args)
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        print("total_train_batch_size: ", total_train_batch_size)
        #quit()

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                print("inputs: ", inputs)
                quit()

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    """
