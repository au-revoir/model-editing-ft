import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from generate import generate_fast, set_generate_seed

class EditDataset:
    def __init__(self, model, tokenizer, prepended_words, args):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.prompt_rewrite = args.prompt_rewrite
        self.prompt_paraphrase_type = args.prompt_paraphrase_type
        self.prepended_words = prepended_words
        self.paraphrase_length_params = [[5, 5], [10, 10]]
        self.prompt_neighborhood_type = args.prompt_neighborhood_type
        self.include_consistency_text = args.include_consistency_text

        self.load_data()
        
        if self.include_consistency_text:
            with open("data/wiki/wiki.json") as fp:
                self.wiki_articles = json.load(fp)

        if args.dataset_name == "zsre":
            self.prepare_zsre_data()
        elif args.dataset_name == "counterfact":
            self.prepare_counterfact_data()
        else:
            raise NotImplementedError
    
    def prepare_zsre_data(self):
        self.data = []
        for record in self.raw_data:
            subject = record["subject"]
            ans_toks = self.tokenizer(" " + record["loc_ans"])["input_ids"]
            item = {"requested_rewrite": {"prompt": record["src"].replace(subject, "{}"), "target_new": {"str": record["answers"][0]},
                                          "target_true": {"str": record["alt"]}, "subject": subject},
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + self.tokenizer.decode(ans_toks[:i]),
                            "target": self.tokenizer.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "random_neighborhood_prompts": [{"prompts": j["prompts"], "target_new": "", "target_true": " ".join(j["target_true"].split()[:35])}
                                                    for j in record["random_neighborhoods"]],
                    "similar_neighborhood_prompts": record["similar_neighborhoods"]
                    }
            self.data.append(item)

    def prepare_counterfact_data(self):
        self.data = []
        for idx, record in enumerate(self.raw_data):
            requested_rewrite = record["requested_rewrite"]
            subject = requested_rewrite["subject"]
            item = {"requested_rewrite": {"prompt": requested_rewrite["prompt"], "target_new": {"str": requested_rewrite["target_new"]["str"], "id": requested_rewrite["target_new"]["id"]},
                                          "target_true": {"str": requested_rewrite["target_true"]["str"], "id": requested_rewrite["target_true"]["id"]}, "subject": subject},
                    "paraphrase_prompts": record["paraphrase_prompts"],
                    "neighborhood_prompts": [{"prompt": i, "target_new": requested_rewrite["target_new"]["str"],
                                              "target_true": requested_rewrite["target_true"]["str"]} for i in record["neighborhood_prompts"]],
                    "random_neighborhood_prompts": record["random_neighborhoods"],
                    "similar_neighborhood_prompts": record["similar_neighborhood_prompts"],
                    }
            if self.include_consistency_text:
                item["consistency_texts"] = [j["text"] for j in self.wiki_articles[idx]]
            self.data.append(item)

    def load_data(self):
        with open(self.data_path) as fp:
            self.raw_data = json.load(fp)
        print("Length of raw data: ", len(self.raw_data))

    def get_dataset(self):
        data_modified = []
        for i in tqdm(self.data):
            all_prompts = []
            all_chosen = []
            all_rejected = []

            requested_rewrite = i["requested_rewrite"]
            subject = requested_rewrite["subject"]

            if self.prompt_rewrite == True:
                rewrite_prompt = [requested_rewrite["prompt"].format(subject)]
                rewrite_chosen_target = [" " + requested_rewrite["target_new"]["str"]] 
                rewrite_rejected_target = [" " + requested_rewrite["target_true"]["str"]] 
                all_prompts.extend(rewrite_prompt)
                all_chosen.extend(rewrite_chosen_target)
                all_rejected.extend(rewrite_rejected_target)

            if self.prompt_paraphrase_type == "generated_prepended_examples":
                if self.prepended_words == None:
                    paraphrase_prompts = [x + ". " + requested_rewrite["prompt"].format(subject) for length, n_gen in self.paraphrase_length_params
                                          for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=n_gen,  max_out_len=length)]
                else:
                    paraphrase_prompts = [x + requested_rewrite["prompt"].format(subject) for x in self.prepended_words["text"]]
                chosen_paraphrase_targets = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_targets = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_targets)
                all_rejected.extend(rejected_paraphrase_targets)

            elif self.prompt_paraphrase_type == "gold_examples":
                paraphrase_prompts = [paraphrase for paraphrase in i["paraphrase_prompts"]]
                chosen_paraphrase_targets = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_targets = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)
                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_targets)
                all_rejected.extend(rejected_paraphrase_targets)

            elif self.prompt_paraphrase_type == "no_examples":
                paraphrase_prompts = []
                chosen_paraphrase_answers = []
                rejected_paraphrase_answers = []
            else:
                raise NotImplementedError

            set_generate_seed(seed=42)
            neighborhood_prompts = []
            chosen_neighborhood_targets = []
            rejected_neighborhood_targets = []

            if self.prompt_neighborhood_type == "similar_examples":
                for neighbor in i["similar_neighborhood_prompts"]: 
                    neighborhood_prompts.append(neighbor["prompts"])
                    chosen_neighborhood_targets.append(" " + neighbor["target_true"])
                    rejected_neighborhood_targets.append(" " + neighbor["target_new"])

            elif self.prompt_neighborhood_type == "random_examples":
                for neighbor in i["random_neighborhood_prompts"]: 
                    neighborhood_prompts.append(neighbor["prompts"])
                    chosen_neighborhood_targets.append(" " + neighbor["target_true"])
                    rejected_neighborhood_targets.append(" " + neighbor["target_new"])

            elif self.prompt_neighborhood_type == "gold_examples":
                    neighborhood_prompts = [prompt["prompt"] for prompt in i["neighborhood_prompts"]]
                    chosen_neighborhood_targets = [" " + requested_rewrite["target_true"]["str"]] 
                    rejected_neighborhood_targets = [" " + requested_rewrite["target_new"]["str"]] 

            elif self.prompt_neighborhood_type == "no_examples":
                    neighborhood_prompts = []
                    chosen_neighborhood_answers = []
                    rejected_neighborhood_answers = []
            else:
                raise NotImplementedError

            all_prompts.extend(neighborhood_prompts)
            all_chosen.extend(chosen_neighborhood_targets)
            all_rejected.extend(rejected_neighborhood_targets)

            if self.include_consistency_text:
                consistency_texts = i["consistency_texts"] + [self.tokenizer.eos_token] * (len(all_prompts) - len(i["consistency_texts"]))
                sample_list = [{"prompt": all_prompts[idx], "chosen": all_chosen[idx], 
                                "rejected": all_rejected[idx], "consistency_text": consistency_texts[idx]} 
                                for idx in range(len(all_prompts))]
            else:
                sample_list = [{"prompt": all_prompts[idx], "chosen": all_chosen[idx], 
                                "rejected": all_rejected[idx]} 
                                for idx in range(len(all_prompts))]
            data_modified.extend(sample_list)
        return data_modified
