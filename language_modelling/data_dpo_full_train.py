import json
import random
from tqdm import tqdm
from transformers import GenerationConfig
from torch.utils.data import Dataset
from generate import generate_fast, set_generate_seed


random.seed(42)

class DPODataset:
    def __init__(self, model_name, model, tokenizer, path, script_args):
        self.model = model
        #self.model.generation_config.pad_token_id = tokenizer.eos_token_id
        #self.generation_config, unused_kwargs = GenerationConfig.from_pretrained(model_name, max_new_tokens=25, temperature=0.9, do_sample=True, 
        #                                                          top_k=50, pad_token_id=tokenizer.eos_token_id, return_unused_kwargs=True)
        self.tokenizer = tokenizer
        self.path = path
        self.prompt_rewrite = script_args.prompt_rewrite
        self.prompt_paraphrase_type = script_args.prompt_paraphrase_type
        self.prompt_neighborhood = script_args.prompt_neighborhood
        self.prompt_neighborhood_type = script_args.prompt_neighborhood_type
        self.num_neighborhood_prompts = script_args.num_neighborhood_prompts
        self.generated_paraphrase = script_args.generated_paraphrase
        self.load_data()
        if script_args.dataset_name == "zsre":
            self.prepare_zsre_data()
        elif script_args.dataset_name == "counterfact":
            self.prepare_counterfact_data()
        else:
            raise NotImplementedError
        #self.length_params = [[10, 50], [20, 50]]
        #self.length_params = [[5, 50], [10, 50]]
        #self.length_params = [[5, 25], [10, 50]]
        self.paraphrase_length_params = [[5, 5], [10, 10]]
        #self.paraphrase_length_params = [[10, 5]]
        #self.paraphrase_length_params = [[10, 1]]
        #self.neighborhood_length_params = [[10, 25]] #[[length, num_sequences]]
        #self.length_params = [[5, 35], [10, 60]]
        #self.length_params = [[5, 100]]
        #self.length_params = [[5, 4], [10, 5]]
        print("paraphrase_length_params: ", self.paraphrase_length_params)

    def prepare_zsre_data(self):
        self.data = []
        for record in self.raw_data:
            item = {"requested_rewrite": {"prompt": record["requested_rewrite"]["prompt"], "target_new": {"str": record["alt"]}, 
                                          "target_true": {"str": record["answers"][0]}, "subject": record["subject"]},
                    "paraphrase_prompts": record["rephrase"], 
                    "neighborhood_prompts": {"prompt": [record["loc"].replace("nq question: ", "")], "target_new": record["alt"], "target_true": record["loc_ans"]}
                    }
            self.data.append(item)

    def prepare_counterfact_data(self):
        self.data = []
        for record in self.raw_data:
            requested_rewrite = record["requested_rewrite"]
            subject = requested_rewrite["subject"]
            item = {"requested_rewrite": {"prompt": requested_rewrite["prompt"], "target_new": {"str": requested_rewrite["target_new"]["str"]}, 
                                          "target_true": {"str": requested_rewrite["target_true"]["str"]}, "subject": subject},
                    "paraphrase_prompts": record["paraphrase_prompts"], 
                    #"starling_generated_paraphrase_prompts": record["paraphrase_prompts_icl_starling"],
                    "neighborhood_prompts": [{"prompt": i, "target_new": requested_rewrite["target_new"]["str"], 
                                              "target_true": requested_rewrite["target_true"]["str"]} for i in record["neighborhood_prompts"]],
                    "random_neighborhood_prompts": record["random_neighborhoods"],
                    "similar_neighborhood_prompts": record["similar_neighborhood_prompts"],
                    "similar_neighborhood_prompts_with_icl": record["icl_with_similar_neighborhood_prompts"],
                    #"starling_generated_neighborhood_prompts": record["neighborhood_prompts_icl_starling"]
                    #"similar_neighborhood_prompts_with_icl_with_true_labels_only": record["icl_with_similar_neighborhood_prompts_true_targets_only"],
                    #"paraphrase_prompts_from_icl": record["paraphrase_prompts_icl"]
                    }
            self.data.append(item)

        
    def load_data(self):
        with open(self.path) as fp:
            self.raw_data = json.load(fp)#[:2000]
        print("Length of raw data: ", len(self.raw_data))

    def get_dataset(self):
        data_modified = []
        #print(data[0])
        #quit()
        for i in tqdm(self.data):
            #print(i)
            all_prompts = []
            all_chosen = []
            all_rejected = []

            requested_rewrite = i["requested_rewrite"]
            subject = requested_rewrite["subject"]

            """
            if self.prompt_neighborhood == True:
                num_neighborhood_prompts = len(i["new_neighborhood_prompts"])
            else:
                num_neighborhood_prompts = 1
            """
            lm_prompts = []
            
            if self.prompt_rewrite == True:
                original_rewrite_prompt = [requested_rewrite["prompt"].format(subject)] #* num_neighborhood_prompts
                original_rewrite_chosen = [" " + requested_rewrite["target_new"]["str"]] #* num_neighborhood_prompts
                original_rewrite_rejected = [" " + requested_rewrite["target_true"]["str"]] #* num_neighborhood_prompts
                all_prompts.extend(original_rewrite_prompt)
                all_chosen.extend(original_rewrite_chosen)
                all_rejected.extend(original_rewrite_rejected)

                lm_prompts.append(requested_rewrite["prompt"].format(subject) + " " + requested_rewrite["target_new"]["str"])
            #"""
            if self.prompt_paraphrase_type == "generated_prepended_examples":
                #length_params = [[10, 50], [20, 50]]
                #length_params = [[5, 50], [10, 50]]
                #length_params = [[5, 25], [10, 25]]
                #length_params = [[10, 2], [20, 1]]
                paraphrase_prompts = [x + ". " + requested_rewrite["prompt"].format(subject) for length, n_gen in self.paraphrase_length_params 
                                      for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=n_gen,  max_out_len=length)]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                #paraphrase_prompts = [x + ". " + requested_rewrite["prompt"].format(subject) for x in paraphrase_generated_prompts]
                #print("paraphrase_prompts: ", paraphrase_prompts)
                #quit()
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])
                #lm_prompts.extend(["dummy"] * len(paraphrase_prompts))  


            elif self.prompt_paraphrase_type == "original_examples":
                paraphrase_prompts = [paraphrase for paraphrase in i["paraphrase_prompts"]]
                #paraphrase_prompts = [paraphrase for paraphrase in i["new_paraphrase_prompts"]]
                #paraphrase_prompts = [paraphrase for paraphrase in i["generation_prompts"][:5]]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])
                #lm_prompts.extend(["dummy"] * len(paraphrase_prompts))
                #print("lm_prompts: ", lm_prompts)
                #quit()
            elif self.prompt_paraphrase_type == "starling_generated_examples":
                #paraphrase_prompts = [paraphrase for paraphrase in i["starling_generated_paraphrase_prompts"]]
                paraphrase_prompts = i["starling_generated_paraphrase_prompts"]#[:5]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                #print("paraphrase_prompts: ", paraphrase_promptsI) 
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])

            elif self.prompt_paraphrase_type == "no_examples":
                neighborhood_prompts = []
                chosen_neighborhood_answers = []
                rejected_neighborhood_answers = []
            else:
                raise NotImplementedError
            #"""

            #print("paraphrase_prompts: ", paraphrase_prompts)
            #quit()
            if self.prompt_neighborhood == True:
                set_generate_seed()
                neighborhood_prompts = []
                chosen_neighborhood_answers = []
                rejected_neighborhood_answers = []
                if self.prompt_neighborhood_type == "similar_examples_with_generation":
                    num_neighborhood_prompts = len(i["similar_neighborhood_prompts"])
                    generated_prompts = [x for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=num_neighborhood_prompts,  max_out_len=10, reset_seed=False)]

                    for neighbor in i["similar_neighborhood_prompts"]:
                        neighborhood_prompts.append(neighbor["prompts"])
                    neighborhod_generated_prompts = [x + ". " + neighborhood_prompts[k] for k in range(len(neighborhood_prompts))
                                         for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=1,  max_out_len=10, reset_seed=False)]
                    neighborhood_prompts.extend(neighborhod_generated_prompts)
                    chosen_neighborhood_answers = [" " + neighbor["target_true"]] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [" " + neighbor["target_true"]] * len(neighborhood_prompts)
                    #print(neighborhood_prompts)    
                    #quit()
                elif self.prompt_neighborhood_type == "similar_examples":
                    for neighbor in i["similar_neighborhood_prompts"]: #There would be 10 random neighborhood prompts with respective target_true and target_new 
                        neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "similar_examples_with_icl":
                    for neighbor in i["similar_neighborhood_prompts_with_icl"]: #There would be 10 random neighborhood prompts with respective target_true and target_new 
                        neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])
                        
                elif self.prompt_neighborhood_type == "similar_examples_with_icl_true_targets_only":
                    for neighbor in i["similar_neighborhood_prompts_with_icl_with_true_labels_only"]: #There would be 10 random neighborhood prompts with respective target_true and target_new 
                        neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "starling_generated_examples":
                    neighborhood_prompts = i["starling_generated_neighborhood_prompts"][:20]
                    chosen_neighborhood_answers = [" " + requested_rewrite["target_true"]["str"]] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [" " + requested_rewrite["target_new"]["str"]] * len(neighborhood_prompts)


                elif self.prompt_neighborhood_type == "random_examples":
                    for neighbor in i["random_neighborhood_prompts"]: #There would be 10 or 20 random neighborhood prompts with respective target_true and target_new 

                        neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "original_examples":
                    neighborhood_prompts = [prompt["prompt"] for prompt in i["neighborhood_prompts"]]#[:len(i["neighborhood_prompts"])//2]
                    chosen_neighborhood_answers = [" " + requested_rewrite["target_true"]["str"]] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [" " + requested_rewrite["target_new"]["str"]] * len(neighborhood_prompts)

                elif self.prompt_neighborhood_type == "generated_examples":
                    #tokenizer.decode(model.generate(max_new_tokens=40, do_sample=True, top_k=0, temperature=0.6, input_ids=tokenizer_text("The mother tongue of is French").to("cuda:2"))[0])
                    neighborhood_ids = self.tokenizer.batch_encode_plus([requested_rewrite["target_true"]["str"]#requested_rewrite["prompt"].replace("{}", "").replace("  ", " ") 
                                                                         for _ in range(20)], 
                                                      return_tensors="pt", padding=True)
                    #print(neighborhood_ids)
                    neighborhood_prompts = self.tokenizer.batch_decode(self.model.generate(input_ids=neighborhood_ids.input_ids.to(self.model.device),
                                                                                      attention_mask=neighborhood_ids.attention_mask.to(self.model.device),
                                                                  generation_config=self.generation_config 
                                                                  ), skip_special_tokens=True) 
                                            #for length, n_gen in self.neighborhood_length_params for _ in range(n_gen)]
                    chosen_neighborhood_answers = [self.tokenizer.eos_token] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [self.tokenizer.eos_token] * len(neighborhood_prompts)
                    #print("neighborhood_prompts: ", neighborhood_prompts)
                    #quit()
                elif self.prompt_neighborhood_type == "no_examples":
                    neighborhood_prompts = []
                    chosen_neighborhood_answers = []
                    rejected_neighborhood_answers = []
                else:
                    raise NotImplementedError

                if self.num_neighborhood_prompts != -1:
                    total_neighborhood_prompts = len(neighborhood_prompts)
                    num_prompts_to_be_selected = min(self.num_neighborhood_prompts, total_neighborhood_prompts)
                    neighborhood_indices_to_select = random.randint(0, num_prompts_to_be_selected - 1)

                    neighborhood_prompts = [neighborhood_prompts[idx] for idx in neighborhood_indices_to_select]
                    chosen_neighborhood_answers = [chosen_neighborhood_answers[idx] for idx in neighborhood_indices_to_select]
                    rejected_neighborhood_answers = [rejected_neighborhood_answers[idx] for idx in neighborhood_indices_to_select]

                all_prompts.extend(neighborhood_prompts)
                all_chosen.extend(chosen_neighborhood_answers)
                all_rejected.extend(rejected_neighborhood_answers)
                lm_prompts.extend([x + c for x, c in zip(neighborhood_prompts, chosen_neighborhood_answers)])
                #lm_prompts.extend([self.tokenizer.eos_token] * len(neighborhood_prompts))
                #lm_prompts.extend(["Dummy"] * len(neighborhood_prompts))

            #print("len(all_prompts): ", len(all_prompts))
            #print("len(all_chosen): ", len(all_chosen))
            #print("len(all_rejected): ", len(all_rejected))
            #print("len(lm_prompts): ", len(lm_prompts))
            #print("len(all_prompts): ", len(all_prompts))
            sample_list = [{"prompt": all_prompts[idx], "chosen": all_chosen[idx], "rejected": all_rejected[idx], "lm_prompt": lm_prompts[idx]} for idx in range(len(all_prompts))]
            #print(sample_list)
            #quit()
            #yield sample
            data_modified.extend(sample_list)
        return data_modified

class FinetuneDataset(Dataset):
    def __init__(self, data, rewrite_prompt, paraphrase_prompt, neighborhood_prompt, tokenizer):
        self.data = data
        #self.prompt_column = prompt_column
        self.rewrite_prompt = rewrite_prompt
        self.paraphrase_prompt = paraphrase_prompt
        self.neighborhood_prompt = neighborhood_prompt
        #print("self.prompt_column: ", self.prompt_column)
        print("self.rewrite_prompt: ", self.rewrite_prompt)
        print("self.paraphrase_prompt: ", self.paraphrase_prompt)
        print("self.neighborhood_prompt: ", self.neighborhood_prompt)
        self.tokenizer = tokenizer
        self._prepare_data()

    def _prepare_data(self):
        self.data_prepared = []
        for i in self.data:
            target_new = i["requested_rewrite"]["target_new"]["str"]
            target_true = i["requested_rewrite"]["target_true"]["str"]
            subject = i["requested_rewrite"]["subject"]
            if self.rewrite_prompt == True:
                if self.method == "ft":
                    prompt = i["requested_rewrite"]["prompt"].format(subject) + " " + target_new
                elif self.method == "dpo":
                    prompt = {"prompt": i["requested_rewrite"]["prompt"].format(subject) + " ", "chosen": target_new, "rejected": target_true}
                self.data_prepared.append(prompt)
            if self.paraphrase_prompt == True:
                if self.method == "ft":
                    prompt = [paraphrase + " " + target_new for paraphrase in i["paraphrase_prompts"]]
                elif self.method == "dpo":
                    prompt = {"prompt": i["requested_rewrite"]["prompt"].format(subject) + " ", "chosen": target_new, "rejected": target_true}
                self.data_prepared.extend(prompt)
            if self.neighborhood_prompt == True:
                prompt = [neighborhood + " " + target_true for neighborhood in i["neighborhood_prompts"]]
                self.data_prepared.extend(prompt)
            
    def __len__(self):
        return len(self.data_prepared)

    def __getitem__(self, idx):
        #target_true = self.data[idx]["requested_rewrite"]["target_true"]["str"]
        #target_new = self.data[idx]["requested_rewrite"]["target_new"]["str"]
        """ 
        if self.prompt_column == "rewrite":
            subject = self.data[idx]["requested_rewrite"]["subject"]
            prompt = self.data[idx]["requested_rewrite"]["prompt"].replace(subject) + " " + target_new
        elif self.prompt_column == "paraphrase":
            prompt = [paraphrase + " " + target_new for paraphrase in self.data[idx]["paraphrase_prompts"]]
        elif self.prompt_column == "neighborhood":
            prompt = [neighborhood + " " + target_true for neighborhood in self.data[idx]["neighborhood_prompts"]]
        else:
            raise NotImplementedError
        print("prompt: ", prompt)
        quit()
        """
        #print("idx: ", idx)
        #print(self.data_prepared[idx])
        #print(self.data_prepared[:20])
        #quit()
        tokenized_inputs = self.tokenizer(self.data_prepared[idx], padding="max_length", max_length=150, return_tensors="pt")
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        #print("tokenized_inputs: ", tokenized_inputs)
        return tokenized_inputs#, self.data_prepared[idx]
