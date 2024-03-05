from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig
from transformers import TrainingArguments

def initialize_training_args(args):
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="run_{}".format(args.model_name),
        num_train_epochs=args.num_epochs,
        dataloader_num_workers=24,
        data_seed=42
    )
    return training_args

def get_peft_config(args):
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_config

@dataclass
class ScriptArguments:
    """
    The arguments for the LM and/or DPO training script.
    """
    prompt_rewrite: Optional[bool] = field(default=False),
    prompt_paraphrase_type: Optional[str] = field(default=None),
    prompt_neighborhood_type: Optional[str] = field(default=None),
    generated_prepended_words_path: Optional[str] = field(default=None)
        
    alpha_loss: Optional[float] = field(default=None),
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    mask_prompt: Optional[bool] = field(default=False) 
    include_consistency_text: Optional[bool] = field(default=False)
    consistency_loss_factor: Optional[float] = field(default=0.0)

    model_name: Optional[str] = field(default="EleutherAI/gpt-j-6B", metadata={"help": "the location of the SFT model name or path"})
    dataset_name: Optional[str] = field(default=None),
    num_epochs: Optional[int] = field(default=2),
    data_size: Optional[int] = field(default=None),
    data_path: Optional[str] = field(default="None", metadata={"help": "dataset path for evaluation and editing with j and k columns"})

    learning_rate: Optional[float] = field(default=4e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16*4, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8*4, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"}) 
    max_length: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"}) 
    max_steps: Optional[int] = field(default=80, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=100000, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="no")
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
