import json
from torch.utils.data import Dataset
from generate import generate_fast

class CFDataset(Dataset):
    def __init__(self, args, model, tokenizer):
        self._load_data(args.path)
        self.prompt_rewrite = args.prompt_rewrite
        self.generated_paraphrase = args.generated_paraphrase
        self.prompt_neighborhood_type = args.prompt_neighborhood_type

        self.model = model
        self.tokenizer = tokenizer

    def _load_data(self, path)
        with open(path, "r") as fp:
            self.data = json.load(fp)

    def __len__(self):
        return len(self.data)

    def _prepare_data(self):
        self.data_modified = []

        for i in tqdm(self.data):
            all_prompts = []
            all_chosen = []
            all_rejected = []
            lm_prompts = []

            requested_rewrite = i["requested_rewrite"]
            subject = requested_rewrite["subject"]

            if self.prompt_rewrite == True:
                original_rewrite_prompt = [requested_rewrite["prompt"].format(subject)] #* num_neighborhood_prompts
                original_rewrite_chosen = [" " + requested_rewrite["target_new"]["str"]] #* num_neighborhood_prompts
                original_rewrite_rejected = [" " + requested_rewrite["target_true"]["str"]] #* num_neighborhood_prompts

                all_prompts.extend(original_rewrite_prompt)
                all_chosen.extend(original_rewrite_chosen)
                all_rejected.extend(original_rewrite_rejected)
                lm_prompts.append(requested_rewrite["prompt"].format(subject) + " " + requested_rewrite["target_new"]["str"])
            
            if self.generated_paraphrase == True:
                paraphrase_prompts = [x + ". " + requested_rewrite["prompt"].format(subject) for length, n_gen in self.paraphrase_length_params
                                      for x in generate_fast(self.model, self.tokenizer, ["<|endoftext|>"], n_gen_per_prompt=n_gen,  max_out_len=length)]
                chosen_paraphrase_answers = [" " + requested_rewrite["target_new"]["str"]] * len(paraphrase_prompts)
                rejected_paraphrase_answers = [" " + requested_rewrite["target_true"]["str"]] * len(paraphrase_prompts)

                all_prompts.extend(paraphrase_prompts)
                all_chosen.extend(chosen_paraphrase_answers)
                all_rejected.extend(rejected_paraphrase_answers)
                lm_prompts.extend([x + " " + requested_rewrite["target_new"]["str"] for x in paraphrase_prompts])

            #set_generate_seed()
            neighborhood_prompts = []
            chosen_neighborhood_answers = []
            rejected_neighborhood_answers = []
            if self.prompt_neighborhood_type == "similar_examples":
                for neighbor in i["similar_neighborhood_prompts"]: #There would be 10 random neighborhood prompts with respective target_true and target_new 
                    neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                    chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                    rejected_neighborhood_answers.append(" " + neighbor["target_new"])

            elif self.prompt_neighborhood_type == "random_examples":
                for neighbor in i["random_neighborhoods"]: #There would be 10 or 20 random neighborhood prompts with respective target_true and target_new 
                    neighborhood_prompts.append(neighbor["prompts"])#[:len(i["neighborhood_prompts"])//2]
                    chosen_neighborhood_answers.append(" " + neighbor["target_true"])
                    rejected_neighborhood_answers.append(" " + neighbor["target_new"])

            elif self.prompt_neighborhood_type == "no_examples":
                neighborhood_prompts = []
                chosen_neighborhood_answers = []
                rejected_neighborhood_answers = []

            else:
                raise NotImplementedError

            all_prompts.extend(neighborhood_prompts)
            all_chosen.extend(chosen_neighborhood_answers)
            all_rejected.extend(rejected_neighborhood_answers)
            lm_prompts.extend([i + c for i, c in zip(neighborhood_prompts, chosen_neighborhood_answers)])

            sample_list = [{"prompt": all_prompts[idx], "chosen": all_chosen[idx], "rejected": all_rejected[idx], "lm_prompts": lm_prompts[idx]} for idx in range(len(all_prompts))]
            self.data_modified.extend(sample_list)

    def __getitem__(self, idx):
        item = self.data[idx]
        print("lm_prompts: ", item["lm_prompts"])
        lm_prompts_tokenized = self.tokenizer.batch_encode_plus(item["lm_prompts"])
        print("lm_prompts_tokenized: ", lm_prompts_tokenized)
        quit()
