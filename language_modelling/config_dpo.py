
from dataclasses import dataclass, field
from typing import Optional

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    start: Optional[int] = field(default=0),
    end: Optional[int] = field(default=0),
    prompt_rewrite: Optional[bool] = field(default=False),
    prompt_paraphrase_type: Optional[str] = field(default=None),
    prompt_neighborhood: Optional[bool] = field(default=False),
    prompt_neighborhood_type: Optional[str] = field(default=None),

    num_neighborhood_prompts: Optional[int] = field(default=None),
    generated_paraphrase: Optional[bool] = field(default=None),

    alpha_loss: Optional[float] = field(default=None),
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name: Optional[str] = field(
        #default="../sft/results/final_checkpoint",
        default="",#/data/local/gg676/DPO/sft_model_checkpoints/eleutherAI_gpt-j-6b/final_checkpoint/final_merged_checkpoint",
        #default="EleutherAI/gpt-j-6B",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name: Optional[str] = field(default=None),
    num_epochs: Optional[int] = field(default=2),
    data_size: Optional[int] = field(default=None),
    data_path: Optional[str] = field(default="None", metadata={"help": "dataset path for evaluation and editing with j and k columns"})
    load_in_4bit: Optional[bool] = field(default=False),

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

    #bf16: Optional[bool] = field(default=True)
    #bf16_full_eval: Optional[bool] = field(default=True)
    #fp16: Optional[bool] = field(default=False)

    lora_alpha: Optional[float] = field(default=16*4, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8*4, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=128, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=80, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=100000, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

