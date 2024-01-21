import json
import gc
import torch

from accelerate import Accelerator
from copy import deepcopy
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import Trainer, set_seed

from trl.trainer.utils import DPODataCollatorWithPadding
from config import ScriptArguments

#from trl import DPOTrainer

def alternate_list(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

"""
def get_dataset(path):
    #with open(path) as fp:
    #    data = json.load(fp)
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset
"""
def get_dataset(path, tokenizer):
    with open(path) as fp:
        data = json.load(fp)
    data_modified = []
    #print(data[0])
    #quit()
    for i in data:
        """
        sample = {"prompt": tokenizer([i["question"], "Boyana Glacier is a part of the continent of"], return_tensors="pt"), 
                 "chosen": tokenizer([i["response_j"], "Europe"], return_tensors="pt"),
                 "rejected": tokenizer([i["response_k"], "Antarctica"], return_tensors="pt")}
        """
        
        """
        sample = {"prompt": [i["question"], "Boyana Glacier is a part of the continent of"],
                  "chosen": [i["response_j"], "Europe"],
                  "rejected": [i["response_k"], "Antarctica"]
                  #"prompt_original": [i["question"]],
                  #"target_true": [i["response_k"]],
                  #"target_new": [i["response_j"]]
                 }
        """
        """
        sample = {"prompt": [i["question"], i["question"]],
                  "chosen": [i["response_j"], i["response_k"]],
                  "rejected": [i["response_k"], i["response_j"]]
                  }
        """
        
        #sample = {"prompt": [i["question"]], "chosen": [i["response_j"]], "rejected": [i["response_k"]]} 

        requested_rewrite = i["requested_rewrite"]
        subject = requested_rewrite["subject"]

        #chosen_neighborhood_prompts = []
        #rejected_neighborhood_prompts = []

        """
        for neighborhood_prompt in i["neighborhood_prompts"]:
            chosen_neighborhood = neighborhood_prompt + " " + requested_rewrite["target_true"]["str"]
            rejected_neighborhood = neighborhood_prompt + " " + requested_rewrite["target_new"]["str"]

            chosen_neighborhood_prompts.append(chosen_neighborhood)
            rejected_neighborhood_prompts.append(rejected_neighborhood_prompts)
        """
        neighborhood_prompts = i["neighborhood_prompts"]#[:len(i["neighborhood_prompts"])//2]
        chosen_neighborhood_answers = [requested_rewrite["target_true"]["str"] + " "] * len(neighborhood_prompts)
        rejected_neighborhood_answers = [requested_rewrite["target_new"]["str"] + " "] * len(neighborhood_prompts)

        original_rewrite_prompt = [requested_rewrite["prompt"].format(subject) + " "] * len(neighborhood_prompts)
        original_rewrite_chosen = [requested_rewrite["target_new"]["str"]] * len(neighborhood_prompts)
        original_rewrite_rejected = [requested_rewrite["target_true"]["str"]] * len(neighborhood_prompts)

        #combined_prompts = alternate_list(original_rewrite_prompt, neighborhood_prompts)
        #combined_chosen = alternate_list(original_rewrite_chosen, chosen_neighborhood_answers)
        #combined_rejected = alternate_list(original_rewrite_rejected, rejected_neighborhood_answers)

        #combined_prompts = neighborhood_prompts + original_rewrite_prompt
        #combined_chosen =  chosen_neighborhood_answers + original_rewrite_chosen
        #combined_rejected = rejected_neighborhood_answers + original_rewrite_rejected
        
        combined_prompts = original_rewrite_prompt + neighborhood_prompts
        combined_chosen =  original_rewrite_chosen + chosen_neighborhood_answers
        combined_rejected = original_rewrite_rejected + rejected_neighborhood_answers
        #requested_rewrite = i["requested_rewrite"]
        #subject = requested_rewrite["subject"]
        #sample = {"prompt": [requested_rewrite["prompt"].format(subject) + " "], "chosen": [requested_rewrite["target_new"]["str"]], "rejected": [requested_rewrite["target_true"]["str"]]}
        sample = {"prompt": combined_prompts, "chosen": combined_chosen, "rejected": combined_rejected}
        #print(sample)
        #quit()
        
        #yield sample
        data_modified.append((sample, i))
    return data_modified

def arrange_prompt(samples):
    return {"prompt": samples["question"],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"]
            }

def initialize_training_args(script_args):
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        #per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        #save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        #evaluation_strategy="steps",
        #eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_gptj-6B",
    )
    return training_args

def get_peft_config():
    peft_config = LoraConfig(
        r=64,#script_args.lora_r,
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
        bias="lora_only",
        task_type="CAUSAL_LM",
    )
    return peft_config


def evaluate_model(model, tokenizer, prompt, targets):
    ppls = []
    for target in targets:
        tgt_len = len(tokenizer.encode(" " + target))
        encodings = tokenizer(prompt + target, return_tensors="pt")
        input_ids = encodings["input_ids"].to("cuda")
        #print("input_ids: ", input_ids)
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

def main(script_args):
    set_seed(42)
    print("script_args.model_name_or_path: ", script_args.model_name_or_path)
    """
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, load_in_4bit=True, 
                                                    low_cpu_mem_usage=True, #device_map={'':torch.cuda.current_device()},
                                                    #device_map={"": Accelerator().local_process_index}, 
                                                    cache_dir="/data/local/gg676/pretrained_models")
    """                                             

    model_base = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            #device_map={"": Accelerator().local_process_index}, 
                                                            device_map={'':torch.cuda.current_device()},
                                                            cache_dir="/data/local/gg676/pretrained_models")
    #model = deepcopy(model_base)
    #state_dict = deepcopy(model_base.state_dict())
    #model_base_copy = deepcopy(model_base)
    #print(state_dict)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            #device_map={"": Accelerator().local_process_index}, 
                                                            device_map={'':torch.cuda.current_device()},
                                                            cache_dir="/data/local/gg676/pretrained_models")

    model_reference = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            #device_map={"": Accelerator().local_process_index}, 
                                                            cache_dir="/data/local/gg676/pretrained_models")
    """
    model_reference = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, #device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                           low_cpu_mem_usage=True,  cache_dir="/data/local/gg676/pretrained_models", load_in_4bit=True)
    """ 
    model_reference.config.use_cache = False
    #model_base = deepcopy(model_reference)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token

    #data_collator = DPODataCollatorWithPadding(tokenizer=tokenizer, max_length=1024)#, max_prompt_length=100)

    data = get_dataset(script_args.data_path, tokenizer)
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

    training_args = initialize_training_args(script_args)

    peft_config = get_peft_config()
    #print("peft_config: ", peft_config)
    #print("task type: ", peft_config.task_type)
    #quit()

    rewrite_success_count = 0
    rewrite_total_count = 0
    paraphrase_success_count = 0
    paraphrase_total_count = 0
    neighborhood_success_count = 0
    neighborhood_total_count = 0

    rewrite_magnitude = 0.
    paraphrase_magnitude = 0.
    neighborhood_magnitude = 0.

    print("training_args: ", training_args)

    #model = deepcopy(model_reference)
    for item, item_original in tqdm(data[:20]):
        from trl import DPOTrainer
        #model_to_be_edited = deepcopy(model)
        #model = deepcopy(model_reference)
        """
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16, load_in_4bit=True, 
                                                        low_cpu_mem_usage=True, #device_map={'':torch.cuda.current_device()},
                                                        device_map={"": Accelerator().local_process_index}, 
                                                        cache_dir="/data/local/gg676/pretrained_models")
        """
        """
        model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.bfloat16,#device_map={'':torch.cuda.current_device()}, 
                                                            device_map={"": Accelerator().local_process_index}, 
                                                         cache_dir="/data/local/gg676/pretrained_models")
        """
        #print(state_dict)
        weights_copy = {}
        with torch.no_grad():
            for name, param in model_base.named_parameters():
                w = get_parameter(model_base, name)
                for n, p in model.named_parameters():
                    if name == n:
                        p[...] = w.detach().clone()
        #model = deepcopy(model_base)
        #model = deepcopy(model_base.detach())
        #state_dict = model_base.state_dict()
        #model = model_base.clone()
        #model.load_state_dict(state_dict)
        #del state_dict
        #print("Before edit")
        #compare_models(model, model_reference)
        #model_reference_copy = deepcopy(model_reference)
        #from trl import DPOTrainer
        #model = deepcopy(model_reference)
        #model_base = deepcopy(model_reference)
        #item["chosen"].append("Europe")
        #item["rejected"].append("Antarctica")
        #prompt = deepcopy(item["prompt"][0])
        #item["prompt"].append("Boyana Glacier is a part of the continent of")

        #print("item: ", item)
        #print("item_original: ", item_original)
        #quit()
        dataset = Dataset.from_dict(item)
        #dataset = dataset.map(batched=True)
        #model = deepcopy(model_reference)
        training_args = initialize_training_args(script_args)


        #tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        #tokenizer.pad_token = tokenizer.eos_token
        
        dpo_trainer = DPOTrainer(
                                model,
                                model_reference,
                                args=training_args,
                                beta=script_args.beta,
                                train_dataset=dataset,
                                #eval_dataset=eval_dataset,
                                tokenizer=tokenizer,
                                peft_config=peft_config,
                                max_prompt_length=script_args.max_prompt_length,
                                max_length=script_args.max_length,
                                
                                #data_collator=data_collator
                            )
        
        """
        dpo_trainer = Trainer(
                                model_reference,
                                args=training_args,
                                #beta=script_args.beta,
                                train_dataset=dataset,
                                #eval_dataset=eval_dataset,
                                tokenizer=tokenizer,
                                #peft_config=peft_config,
                                #max_prompt_length=script_args.max_prompt_length,
                                #max_length=script_args.max_length,

                                #data_collator=data_collator
                            )
        """
        dpo_trainer.train()
        dpo_trainer.model.eval()

        #print("After edit")
        #compare_models(model, model_reference)
        #input_ids_original = item_o
        

        #prompt = item["prompt"][0]
        #target_new = item["chosen"][0]
        #target_true = item["rejected"][0]
        requested_rewrite = item_original["requested_rewrite"]
        rewrite_prompt = requested_rewrite["prompt"].format(requested_rewrite["subject"])

        #input_ids_original = tokenizer.tokenize([rewrite_prompt], return_tensors="pt")["input_ids"].to("cuda")
        #outputs = dpo_trainer.model.generate(input_ids=input_ids_original, max_length=100)
        #decoded_output = tokenizer.decode(outputs[0])
        #print("decoded_output: ", decoded_output)


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
            ppl_neighborhood_target_new, ppl_neighborhood_target_true = evaluate_model(dpo_trainer.model, tokenizer, neighbor, targets)
            prob_neighborhood_target_new, prob_neighborhood_target_true = 1 / ppl_neighborhood_target_new, 1 / ppl_neighborhood_target_true

            if prob_neighborhood_target_new < prob_neighborhood_target_true:
                neighborhood_success_count += 1

            neighborhood_magnitude += prob_neighborhood_target_true - prob_neighborhood_target_new

            neighborhood_total_count += 1

        #del dpo_trainer
        #del state_dict
        #del DPOTrainer
        #del dataset
        #dpo_trainer = None
        #gc.collect()
        
        print("rewrite_success_count: ", rewrite_success_count)
        print("paraphrase_success_count: ", paraphrase_success_count)
        print("neighborhood_success_count: ", neighborhood_success_count)
        #quit()
        """
        #quit()
        test_prompt_1 = "Danielle Darrieux spoke the language"
        test_prompt_2 = "Danielle Darrieux, a native"
        test_prompt_3 = "The mother tongue of Léon Blum is"
        test_prompt_4 = "The native language of Montesquieu is"
        inputs = tokenizer(test_prompt_4, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=50)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("decoded: ", decoded)
        quit()   
        """

    print("rewrite_success_accuracy: ", rewrite_success_count / rewrite_total_count)
    print("paraphrase_success_accuracy: ", paraphrase_success_count / paraphrase_total_count)
    print("neighborhood_success_accuracy: ", neighborhood_success_count / neighborhood_total_count)


    print("rewrite_magnitude: ", rewrite_magnitude / rewrite_total_count)
    print("paraphrase_magnitude: ", paraphrase_magnitude / paraphrase_total_count)
    print("neighborhood_magnitude: ", neighborhood_magnitude / neighborhood_total_count)
    #print(dataset[0])
    #quit()



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
