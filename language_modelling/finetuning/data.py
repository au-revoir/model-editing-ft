from torch.utils.data import Dataset

map_prompt_columns = {"rewrite": "requested_rewrite", "paraphrase": "paraphrase_prompts", "neighborhood": "neighborhood_prompts"}

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
                prompt = i["requested_rewrite"]["prompt"].format(subject) + " " + target_new
                self.data_prepared.append(prompt)
            if self.paraphrase_prompt == True:
                prompt = [paraphrase + " " + target_new for paraphrase in i["paraphrase_prompts"]]
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

