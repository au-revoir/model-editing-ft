import os
import json
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from config import initialize_training_args, get_peft_config, ScriptArguments
from data import EditDataset
from trainer import CustomTrainer
from eval_counterfact import compute_counterfact_predictions
from eval_zsre import compute_zsre_predictions
from utils import load_json, create_incremented_directory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    accelerator = Accelerator()
    set_seed(42)
    model_name = args.model_name

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map={"": Accelerator().local_process_index},
                                                 cache_dir="/data/local/gg676/pretrained_models")
    
    model_reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map={"": Accelerator().local_process_index},
                                                 cache_dir="/data/local/gg676/pretrained_models")
    model.config_use = False
    model_reference.config_use = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.generated_prepended_words_path is not None:
        random_prepend_words = load_json(args.generated_prepended_words_path)
    else:
        random_prepend_words = None

    dataset = EditDataset(model, tokenizer, random_prepend_words, args)
    data_formatted = dataset.get_dataset()
    dataset_formatted = Dataset.from_list(data_formatted)

    #print("\ndataset_formatted: ", dataset_formatted[:4])
    #quit()

    training_args = initialize_training_args(args)

    if model_name == "EleutherAI/gpt-j-6B":
        peft_config = get_peft_config(args)
    else:
        peft_config = None
    
    if accelerator.is_main_process:
        print("Training args: ", training_args)

    trainer = CustomTrainer(model,
                            model_reference,
                            args=training_args,
                            beta=args.beta,
                            train_dataset=dataset_formatted,
                            tokenizer=tokenizer,
                            peft_config=peft_config,
                            max_prompt_length=args.max_prompt_length,
                            max_length=args.max_length,
                            disable_dropout=True,
                            mask_prompt=args.mask_prompt,
                            include_consistency_text=args.include_consistency_text,
                            )
    trainer.train()
    accelerator.wait_for_everyone()
    trainer.model.eval()

    if accelerator.is_main_process:
        if args.dataset_name == "zsre":
            compute_zsre_predictions(trainer.model, tokenizer, dataset.data)
        else:
            compute_counterfact_predictions(trainer.model, tokenizer, dataset.data)

        model_save_path = create_incremented_directory(args.output_dir)
        trainer.model.save_pretrained("{}/{}".format(model_save_path, args.model_name))
        print("Model saved to {}".format(model_save_path))

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
