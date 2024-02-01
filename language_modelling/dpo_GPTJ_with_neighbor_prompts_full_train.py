import os
import json
import gc
import torch
import statistics

from accelerate import Accelerator
from copy import deepcopy
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from transformers import Trainer, set_seed
from peft import PeftModel


from trl.trainer.utils import DPODataCollatorWithPadding
from config_dpo import ScriptArguments
from losses_modified_prompt_softmax_masking import CustomDPOTrainer
#from losses_modified import CustomDPOTrainer
from data_dpo_full_train import DPODataset
from evaluation import CounterFactEval
from generate import set_generate_seed 

from eval_zsre import evaluate_zsre_model 
from eval_counterfact import compute_counterfact_predictions

import globals
#global eval_method
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#from trl import DPOTrainer
torch.autograd.set_detect_anomaly(True)

def load_json(path):
    with open(path) as fp:
        data = json.load(fp)
    return data

def save_json(data, path):
    with open(path, "w") as fp:
        json.dump(data, fp)

def alternate_list(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

def get_dataset(path):
    #with open(path) as fp:
    #    data = json.load(fp)
    dataset = load_dataset("json", data_files=path, split="train")

def arrange_prompt(samples):
    return {"prompt": samples["question"],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"]
            }

def initialize_training_args(script_args):
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        #per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        #max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        #gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        #evaluation_strategy="steps",
        #eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        #weight_decay=script_args.weight_decay,
        optim=script_args.optimizer_type,
        bf16=True,
        #bf16=False,
        #fp16=False,
        #tf32=False,
        #max_length=script_args.max_length,
        remove_unused_columns=False,
        run_name="dpo_gptj-6B",
        num_train_epochs=script_args.num_epochs,
        dataloader_num_workers=24,
        data_seed=42
    )
    return training_args

def get_peft_config():
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
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

def evaluate_model(model, tokenizer, prompt, targets):
    ppls = []
    #prompt = prompt + " "
    #print("prompt: ")
    for target in targets:
        #print("prompt + target: ", prompt + target)
        #quit()
        #print("tokenizer.encode(target): ", tokenizer.encode(target))
        #print("tokenizer(target): ", tokenizer(target))
        tgt_len = len(tokenizer.encode(" " + target))
        encodings = tokenizer(prompt + " " + target, return_tensors="pt")
        input_ids = encodings["input_ids"].to("cuda")
        #print("prompt + target input_ids: ", input_ids)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        #print("target_ids: ", target_ids)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            #print("loss: ", outputs.loss)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
        del outputs
        #print("ppl: ", ppl)
        #print("tgt_len: ", tgt_len)
        #print("encodings: ", encodings)
        #quit()
    #quit()
    return ppls

#def batch_evaluate_model(model, tokenizer, prompt, targets)

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

def decode_prompt_outputs(model, tokenizer, prompt, prompt_type):
    print("\nPrompt type: ", prompt_type)
    tokenizer.padding_side = "left"
    with torch.no_grad():
        input_ids_original = tokenizer(prompt, return_tensors="pt", max_length=30, padding=True).to("cuda")
        outputs = model.generate(**input_ids_original, max_length=50)
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    tokenizer.padding_side = "right"
    print("decoded_output: ", decoded_output)

def main(script_args):
    accelerator = Accelerator()
    #set_seed(42)
    #print("script_args.model_name_or_path: ", script_args.model_name_or_path)
    """
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, load_in_4bit=True, 
                                                    low_cpu_mem_usage=True, #device_map={'':torch.cuda.current_device()},
                                                    #device_map={"": Accelerator().local_process_index}, 
                                                    cache_dir="/data/local/gg676/pretrained_models")
    """                                             
    """
    model_base = AutoModelForCausalLM.from_pretrained("gpt2-xl",torch_dtype=torch.float16, #torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                            #device_map={'':torch.cuda.current_device()},
                                                            cache_dir="/data/local/gg676/pretrained_models")
    """
    #model = deepcopy(model_base)
    #state_dict = deepcopy(model_base.state_dict())
    #model_base_copy = deepcopy(model_base)
    #print(state_dict)

    #model_name = "EleutherAI/gpt-j-6B" # "gpt2-xl"
    model_name = script_args.model_name
    #if script_args.load_in_4bit:
    #    print("Loading {} in 4-bit".format(model_name))
    #print("script_args.load_in_4bit: ", script_args.load_in_4bit)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,#torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                            #device_map={'':torch.cuda.current_device()},
                                                            #low_cpu_mem_usage=True,
                                                            #load_in_4bit=script_args.load_in_4bit,
                                                            #quantization_config=bnb_config,
                                                            cache_dir="/data/local/gg676/pretrained_models")

    model_reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                            #low_cpu_mem_usage=True,
                                                            #load_in_4bit=script_args.load_i`n_4bit,
                                                            #quantization_config=bnb_config,
                                                            cache_dir="/data/local/gg676/pretrained_models")

    #model = PeftModel.from_pretrained(model, "/data/local/gg676/model_editing/saved_models/gpt-j-6B.pt")
    #model_reference = PeftModel.from_pretrained(model_reference, "/data/local/gg676/model_editing/saved_models/gpt-j-6B.pt")
    """
    model_reference = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                           low_cpu_mem_usage=True,  cache_dir="/data/local/gg676/pretrained_models", load_in_4bit=True)
    """ 
    model_reference.config.use_cache = False
    #model_base.config.use_cache = False
    model.config.use_cache = False
    #model_base = deepcopy(model_reference)

    tokenizer = AutoTokenizer.from_pretrained(model_name)#("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DPODataCollatorWithPadding(tokenizer=tokenizer, max_length=128)#, max_prompt_length=100)
    dataset_dpo = DPODataset(model_name, model_reference, tokenizer, script_args.data_path, script_args)
    #if False:
    if os.path.isfile(script_args.data_path[:-5] + "AddedPrependedParaphrase_10k.json"):
        #data_transformed = load_json(script_args.data_path[:-5] + "AddedPrependedParaphrase_4bitModels.json")
        data_transformed = load_json(script_args.data_path[:-5] + "AddedPrependedParaphrase_10k.json")
    else:
        data_transformed = dataset_dpo.get_dataset()


    #for item_original in tqdm(dataset_dpo.data):
        


    #if Accelerator().is_main_process:
        #save_json(data_transformed, script_args.data_path[:-5] + "AddedPrependedParaphrase_10k.json")

    #train_edit_data = [item[0] for item in data]
    #eval_edit_data = [item[1] for item in data]

    #print("train_edit_data[:2]: ", train_edit_data[:2])
    #train_dataset = Dataset.from_list(train_edit_data[:10])
    #print(train_dataset[0])

    #globals.init(eval_edit_data, tokenizer)
    #eval_method = CounterFactEval(data, tokenizer)
    #data = get_dataset(script_args.data_path, tokenizer)
    #print("data: ", data[0])
    #quit()
    #dataset = get_dataset(script_args.data_path)
    #for i in dataset:
    #    print(i)
    #    quit()
    #print(dataset[0])
    #quit()
    #original_columns = dataset.column_names
    #dataset = dataset.map(arrange_prompt, remove_columns=original_columns)
    #print(dataset[0])
    #quit()

    """
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in [i for i in range(5, 45, 1)]
        if "transformer.h.{}.mlp.c_proj".format(layer) in n
    }
    """
    #print(model)

    #print("weights: ", weights)
    #quit()
    #for name, w in weights.items():
    #    print(name)

    #quit()
    """
    for name, w in model.named_parameters():
        w.requires_grad = name in weights
    """
    dataset = Dataset.from_list(data_transformed)
    #print(dataset[0])
    #print(dataset[0][:2])
    #quit()
    training_args = initialize_training_args(script_args)

    peft_config = get_peft_config()



    #print("peft_config: ", peft_config)
    #print("task type: ", peft_config.task_type)
    #quit()
    #eval_edit_data = [item[1] for item in data]
    #print("eval_edit_data[0]: ", eval_edit_data[0])

    rewrite_success_count = 0
    rewrite_total_count = 0
    paraphrase_success_count = 0
    paraphrase_total_count = 0
    neighborhood_success_count = 0
    neighborhood_total_count = 0

    rewrite_magnitude = 0.
    paraphrase_magnitude = 0.
    neighborhood_magnitude = 0.

    if Accelerator().is_main_process:
        print("training_args: ", training_args)
    #dataset = Dataset.from_dict(item)
    #print("dataset inside for loop: ", dataset[:])
    #quit()
    #dataset = dataset.map(batched=True)
    #model = deepcopy(model_reference)
    training_args = initialize_training_args(script_args)

    #print("dataset: ", dataset[:])

    #tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    #tokenizer.pad_token = tokenizer.eos_token
    
    dpo_trainer = CustomDPOTrainer(#DPOTrainer(
                            model,
                            model_reference,
                            args=training_args,
                            beta=script_args.beta,
                            train_dataset=dataset,
                            #eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            peft_config=peft_config,
                            #eval_edit_data=eval_edit_data,
                            max_prompt_length=script_args.max_prompt_length,
                            max_length=script_args.max_length,
                            disable_dropout=True,
                            #disable_dropout=False,
                            #loss_type="ipo",
                            
                            #find_unused_parameters=False, 
                            data_collator=data_collator
                        )
    
    """
    dpo_trainer = Trainer(
                            model,
                            args=training_args,
                            beta=script_args.beta,
                            train_dataset=dataset,
                            #eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            #peft_config=peft_config,
                            #max_prompt_length=script_args.max_prompt_length,
                            max_length=script_args.max_length,

                            #data_collator=data_collator
                        )

    """
    dpo_trainer.train()

    accelerator.wait_for_everyone()
    #dpo_trainer.model.to("cuda")
    dpo_trainer.model.eval()

    if accelerator.is_main_process:
        if "/" in model_name:
            save_model_name = model_name.split("/")[1]
        else:
            save_model_name = model_name
        #print("save_model_name: ", save_model_name)
        #dpo_trainer.model.save_pretrained("/data/local/gg676/model_editing/saved_models/{}.pt".format(save_model_name))

    #quit()
    #print("After edit")
    #compare_models(model, model_reference)
    #input_ids_original = item_o
    if accelerator.is_main_process:   
        compute_counterfact_predictions(dpo_trainer.model, tokenizer, dataset_dpo.data)
        quit()
    #if True:
    #if Accelerator().local_process_index == 0:
        for item_original in tqdm(dataset_dpo.data):
            #prompt = item["prompt"][0]
            #target_new = item["chosen"][0]
            #target_true = item["rejected"][0]
            requested_rewrite = item_original["requested_rewrite"]
            rewrite_prompt = requested_rewrite["prompt"].format(requested_rewrite["subject"])
            """
            input_ids_original = tokenizer([rewrite_prompt], return_tensors="pt")["input_ids"].to("cuda")
            outputs = dpo_trainer.model.generate(input_ids=input_ids_original, max_length=100)
            decoded_output = tokenizer.decode(outputs[0])
            print("decoded_output: ", decoded_output)
            """

            target_new = requested_rewrite["target_new"]["str"]
            target_true = requested_rewrite["target_true"]["str"]
            targets = [target_new, target_true]

            ppl_rewrite_prompt_target_new, ppl_rewrite_prompt_target_true = evaluate_model(dpo_trainer.model, tokenizer, rewrite_prompt, targets)

            prob_rewrite_prompt_target_new, prob_rewrite_prompt_target_true = 1 / ppl_rewrite_prompt_target_new, 1 / ppl_rewrite_prompt_target_true

            if prob_rewrite_prompt_target_new > prob_rewrite_prompt_target_true:
                rewrite_success_count += 1
            rewrite_total_count += 1
            rewrite_magnitude += prob_rewrite_prompt_target_new - prob_rewrite_prompt_target_true
            #print("rewrite_success_count: ", rewrite_success_count)

            for paraphrase in item_original["paraphrase_prompts"]:
                ppl_paraphrase_target_new, ppl_paraphrase_target_true = evaluate_model(dpo_trainer.model, tokenizer, paraphrase, targets)
                prob_paraphrase_target_new, prob_paraphrase_target_true = 1 / ppl_paraphrase_target_new, 1 / ppl_paraphrase_target_true
                
                if prob_paraphrase_target_new > prob_paraphrase_target_true:
                    paraphrase_success_count += 1
                
                paraphrase_magnitude += prob_paraphrase_target_new - prob_paraphrase_target_true

                paraphrase_total_count += 1

            for neighbor in item_original["neighborhood_prompts"]:
                targets = [neighbor["target_new"], neighbor["target_true"]]
                ppl_neighborhood_target_new, ppl_neighborhood_target_true = evaluate_model(dpo_trainer.model, tokenizer, neighbor["prompt"], targets)
                prob_neighborhood_target_new, prob_neighborhood_target_true = 1 / ppl_neighborhood_target_new, 1 / ppl_neighborhood_target_true

                if prob_neighborhood_target_new < prob_neighborhood_target_true:
                    neighborhood_success_count += 1

                neighborhood_magnitude += prob_neighborhood_target_true - prob_neighborhood_target_new

                neighborhood_total_count += 1

            #decode_prompt_outputs(dpo_trainer.model, tokenizer, [rewrite_prompt], prompt_type="rewrite")
        #decode_prompt_outputs(dpo_trainer.model, tokenizer, item_original["paraphrase_prompts"], prompt_type="paraphrase")
        #decode_prompt_outputs(dpo_trainer.model, tokenizer, item_original["neighborhood_prompts"], prompt_type="neighborhood")
        #quit()
        #del dpo_trainer
        #del state_dict
        #del DPOTrainer
        #del dataset
        #dpo_trainer = None
        #gc.collect()

        rewrite_success_accuracy = rewrite_success_count / rewrite_total_count
        paraphrase_success_accuracy = paraphrase_success_count / paraphrase_total_count
        neighborhood_success_accuracy = neighborhood_success_count / neighborhood_total_count

        
        print("rewrite_success_count: ", rewrite_success_count)
        print("paraphrase_success_count: ", paraphrase_success_count)
        print("neighborhood_success_count: ", neighborhood_success_count)


        print("rewrite_total_count: ", rewrite_total_count)
        print("paraphrase_total_count: ", paraphrase_total_count)
        print("neighborhood_total_count: ", neighborhood_total_count)

        print("rewrite_success_accuracy: ", rewrite_success_count / rewrite_total_count)
        print("paraphrase_success_accuracy: ", paraphrase_success_count / paraphrase_total_count)
        print("neighborhood_success_accuracy: ", neighborhood_success_count / neighborhood_total_count)


        print("rewrite_magnitude: ", rewrite_magnitude / rewrite_total_count)
        print("paraphrase_magnitude: ", paraphrase_magnitude / paraphrase_total_count)
        print("neighborhood_magnitude: ", neighborhood_magnitude / neighborhood_total_count)

        #print("Average Score: ", (rewrite_success_accuracy + paraphrase_success_accuracy + neighborhood_success_accuracy) / 3)
        print("Harmonic Mean Score: ", statistics.harmonic_mean([rewrite_success_accuracy, paraphrase_success_accuracy, neighborhood_success_accuracy]))
        #print(dataset[0])
        #quit()

        print(rewrite_success_count / rewrite_total_count, paraphrase_success_count / paraphrase_total_count, neighborhood_success_count / neighborhood_total_count, rewrite_magnitude / rewrite_total_count, paraphrase_magnitude / paraphrase_total_count, neighborhood_magnitude / neighborhood_total_count)



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
