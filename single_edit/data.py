import json
import random
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from torch.utils.data import Dataset
from generate import generate_fast, set_generate_seed

random.seed(42)

class LMDataset:
    def __init__(self, args, model, tokenizer, prepended_words):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.data_start = args.data_start
        self.data_end = args.data_end
        self.prompt_rewrite = args.prompt_rewrite
        self.prompt_paraphrase_type = args.prompt_paraphrase_type
        self.prompt_neighborhood = args.prompt_neighborhood
        self.prompt_neighborhood_type = args.prompt_neighborhood_type
        self.prepended_words = prepended_words
        self.load_data()
        if args.dataset_name == "zsre":
            self.prepare_zsre_data()
        elif args.dataset_name == "counterfact":
            self.prepare_counterfact_data()
        else:
            raise NotImplementedError
        self.prepended_words = prepended_words
        self.paraphrase_length_params = [[5, 5], [10, 10]]

    def prepare_zsre_data(self):
        self.data = []
        for idx, record in enumerate(self.raw_data):
            subject = record["subject"]
            ans_toks = self.tokenizer(" " + record["loc_ans"])["input_ids"]
            case_id = record["case_id"]
            item = {"case_id": case_id,
                    "requested_rewrite": {"prompt": record["src"].replace(subject, "{}"), "target_new": {"str": record["answers"][0]}, 
                                          "target_true": {"str": ""}, "subject": subject},
                    "paraphrase_prompts": [record["rephrase"]], 
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + self.tokenizer.decode(ans_toks[:i]),
                            "target": self.tokenizer.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "random_neighborhood_prompts": record["random_neighborhoods"],
                    "similar_neighborhood_prompts": record["similar_neighborhoods"]
                    }
            self.data.append(item)

    def prepare_counterfact_data(self):
        self.data = []
        for record in self.raw_data:
            requested_rewrite = record["requested_rewrite"]
            subject = requested_rewrite["subject"]
            case_id = record["case_id"]
            item = {"case_id": case_id,
                    "requested_rewrite": {"prompt": requested_rewrite["prompt"], "target_new": {"str": requested_rewrite["target_new"]["str"], "id": requested_rewrite["target_new"]["id"]}, 
                                          "relation_id": requested_rewrite["relation_id"], 
                                          "target_true": {"str": requested_rewrite["target_true"]["str"], "id": requested_rewrite["target_true"]["id"]}, 
                                          "subject": subject},
                    "paraphrase_prompts": record["paraphrase_prompts"], 
                    "neighborhood_prompts": [{"prompt": i, "target_new": requested_rewrite["target_new"]["str"], 
                                              "target_true": requested_rewrite["target_true"]["str"]} for i in record["neighborhood_prompts"]],
                    "random_neighborhood_prompts": record["random_neighborhoods"],
                    "similar_neighborhood_prompts": record["similar_neighborhood_prompts"],
                    "generation_prompts": record["generation_prompts"],
                    "attribute_prompts": record["attribute_prompts"]
                    }
            self.data.append(item)

        
    def load_data(self):
        with open(self.data_path) as fp:
            self.raw_data = json.load(fp)[self.data_start: self.data_end]
        print("Length of raw data: ", len(self.raw_data))

    def get_dataset(self):
        data_modified = []
        for i in tqdm(self.data):
            all_prompts = []
            all_chosen = []
            all_rejected = []

            requested_rewrite = i["requested_rewrite"]
            subject = requested_rewrite["subject"]

            lm_prompts = []
            tags = []
            subject_text = "{} is a".format(subject)
            
            if self.prompt_rewrite == True:
                original_rewrite_prompt = [requested_rewrite["prompt"].format(subject)] 
                original_rewrite_chosen = [" " + requested_rewrite["target_new"]["str"]] 
                original_rewrite_rejected = [" " + requested_rewrite["target_true"]["str"]] 
                all_prompts.extend(original_rewrite_prompt)
                all_chosen.extend(original_rewrite_chosen)
                all_rejected.extend(original_rewrite_rejected)

                lm_prompts.append(requested_rewrite["prompt"].format(subject) + " " + requested_rewrite["target_new"]["str"])
                tags.append(torch.tensor(False))

            if self.prompt_paraphrase_type == "generated_prepended_examples":
                if self.prepended_words == None:
                    paraphrase_prompts = [x + ". " + requested_rewrite["prompt"].format(subject) for length, n_gen in self.paraphrase_length_params 
                                          for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=n_gen,  max_out_len=length)]
                else:
                    paraphrase_prompts = [x + requested_rewrite["prompt"].format(subject) for x in self.prepended_words["text"][:10]] 
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])
                tags.extend([torch.tensor(False)] * len(paraphrase_prompts))

            elif self.prompt_paraphrase_type == "original_examples":
                paraphrase_prompts = [paraphrase for paraphrase in i["paraphrase_prompts"]]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])

            elif self.prompt_paraphrase_type == "starling_generated_examples":
                paraphrase_prompts = i["starling_generated_paraphrase_prompts"]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])

            elif self.prompt_paraphrase_type == "no_examples":
                paraphrase_prompts = []
                chosen_paraphrase_answers = []
                rejected_paraphrase_answers = []
            else:
                raise NotImplementedError

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

                elif self.prompt_neighborhood_type == "similar_examples":
                    for neighbor in i["similar_neighborhood_prompts"][:15]: 
                        neighborhood_prompts.append(neighbor["prompts"])
                        target_true = " ".join(neighbor["target_true"].split()[:35])
                        chosen_neighborhood_answers.append(" " + target_true)
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "similar_examples_with_icl":
                    for neighbor in i["similar_neighborhood_prompts_with_icl"]: 
                        neighborhood_prompts.append(neighbor["prompts"])
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])
                        
                elif self.prompt_neighborhood_type == "similar_examples_with_icl_true_targets_only":
                    for neighbor in i["similar_neighborhood_prompts_with_icl_with_true_labels_only"]: 
                        neighborhood_prompts.append(neighbor["prompts"])
                        chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "starling_generated_examples":
                    neighborhood_prompts = [prompt["prompt"] for prompt in i["starling_generated_neighborhood_prompts"]]
                    chosen_neighborhood_answers = [" " + target["target_true"] for target in i["starling_generated_neighborhood_prompts"]]
                    rejected_neighborhood_answers = [" " + target["target_new"] for target in i["starling_generated_neighborhood_prompts"]] 


                elif self.prompt_neighborhood_type == "random_examples":
                    for neighbor in i["random_neighborhood_prompts"][:15]: 
                        target_true = " ".join(neighbor["target_true"].split()[:35])
                        neighborhood_prompts.append(neighbor["prompts"])
                        chosen_neighborhood_answers.append(" " + target_true)
                        rejected_neighborhood_answers.append(" " + neighbor["target_new"])

                elif self.prompt_neighborhood_type == "original_examples":
                    neighborhood_prompts = [prompt["prompt"] for prompt in i["neighborhood_prompts"]]
                    chosen_neighborhood_answers = [" " + requested_rewrite["target_true"]["str"]] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [" " + requested_rewrite["target_new"]["str"]] * len(neighborhood_prompts)

                elif self.prompt_neighborhood_type == "generated_examples":
                    neighborhood_ids = self.tokenizer.batch_encode_plus([requested_rewrite["target_true"]["str"]
                                                                         for _ in range(20)], 
                                                      return_tensors="pt", padding=True)
                    neighborhood_prompts = self.tokenizer.batch_decode(self.model.generate(input_ids=neighborhood_ids.input_ids.to(self.model.device),
                                                                                      attention_mask=neighborhood_ids.attention_mask.to(self.model.device),
                                                                  generation_config=self.generation_config 
                                                                  ), skip_special_tokens=True) 
                    chosen_neighborhood_answers = [self.tokenizer.eos_token] * len(neighborhood_prompts)
                    rejected_neighborhood_answers = [self.tokenizer.eos_token] * len(neighborhood_prompts)

                elif self.prompt_neighborhood_type == "no_examples":
                    neighborhood_prompts = []
                    chosen_neighborhood_answers = []
                    rejected_neighborhood_answers = []

                else:
                    raise NotImplementedError

                all_prompts.extend(neighborhood_prompts)
                all_chosen.extend(chosen_neighborhood_answers)
                all_rejected.extend(rejected_neighborhood_answers)
                lm_prompts.extend([x + c for x, c in zip(neighborhood_prompts, chosen_neighborhood_answers)])
                tags.extend([torch.tensor(True)] * len(neighborhood_prompts))

            sample_list = [{"prompt": all_prompts[idx], "chosen": all_chosen[idx], "lm_prompt": lm_prompts[idx]} for idx in range(len(all_prompts))]
            data_modified.append(sample_list)
        return data_modified
