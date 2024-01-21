import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed
from data import CFDataset
from utils import save_json

#PROMPT_TEMPLATE = "<s>[INST] Generate 10 paraphrases. Do NOT write the answer to the question. Only write the paraphrases: {} {}[/INST]"
#PROMPT_TEMPLATE = "<s>[INST] Generate 10 paraphrases. Write the paraphrases in the same format as: {} {}[/INST]"

class InferenceDDP:
    def __init__(self, gpu_id, model, tokenizer, output_save_path, max_generated_token_length=1024):
        self.gpu_id = gpu_id
        self.model = model#.to(gpu_id)
        self.tokenizer = tokenizer
        self.output_save_path = output_save_path
        self.output_file_name = "results_GPU{}.json".format(str(self.gpu_id))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id
        self.max_generated_token_length = max_generated_token_length

    def prepare_dataloader(self, dataset, batch_size=2):
        return DataLoader(dataset, batch_size=batch_size, 
                              pin_memory=False, shuffle=False, 
                              sampler =DistributedSampler(dataset),
                              )

    def _run_generate_text(self, tokenized_inputs):
        set_seed(0)
        outputs = self.model.generate(input_ids=tokenized_inputs.input_ids.to(self.gpu_id), 
                                     attention_mask=tokenized_inputs.attention_mask.to(self.gpu_id), 
                                     max_length=self.max_generated_token_length, 
                                     pad_token_id=self.pad_token_id, do_sample=False)#, temperature=0.6)
        output_decoded_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_decoded_text

    def run_icl(self, val_dataloader):
        generated_results = []

        for batch in tqdm(val_dataloader):
            #print(batch)
            text = batch["text"]
            #text = PROMPT_TEMPLATE.format(batch["requested_rewrite"]["prompt"][0].format(batch["requested_rewrite"]["subject"][0]),  batch["requested_rewrite"]["target_true"][0])
            #print("\ntext: ", text)
            #quit()
            tokenized_inputs = self.tokenizer.batch_encode_plus(batch["text"],
                                                        return_tensors="pt", 
                                                        padding="longest")
            output_decoded_text = self._run_generate_text(tokenized_inputs)
            #print("output_decoded_text: ", output_decoded_text)
            for i in range(len(output_decoded_text)):
                item = {"case_id": batch["case_id"][i].item(), 
                        "requested_rewrite": {"prompt": batch["requested_rewrite"]["prompt"][i],
                        "subject": batch["requested_rewrite"]["subject"][i],
                        "relation_id": batch["requested_rewrite"]["relation_id"][i]},
                        "generated_neighborhoods": output_decoded_text[i]}
                generated_results.append(item)
            #print(item)
        return generated_results

    def evaluate(self, val_data, icl_type):
        val_dataset = CFDataset(val_data, icl_type)
        #print(val_dataset[0])
        
        val_dataloader = self.prepare_dataloader(val_dataset, batch_size=32)
        generated_val_results = self.run_icl(val_dataloader)
        save_path = "{}/{}".format(self.output_save_path, self.output_file_name)
        save_json(save_path, generated_val_results)
