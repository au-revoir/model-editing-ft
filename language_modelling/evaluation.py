import torch

class CounterFactEval:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.rewrite_success_count = 0
        self.rewrite_total_count = 0

        self.paraphrase_success_count = 0
        self.paraphrase_total_count = 0
        self.neighborhood_success_count = 0
        self.neighborhood_total_count = 0

        self.rewrite_magnitude = 0.
        self.paraphrase_magnitude = 0.
        self.neighborhood_magnitude = 0.
    
    def evaluate_model(self, model, prompt, targets):
        ppls = []
        #prompt = prompt + " "
        #print("prompt: ")
        for target in targets:
            #print("prompt + target: ", prompt + target)
            #print("tokenizer.encode(target): ", tokenizer.encode(target))
            #print("tokenizer(target): ", tokenizer(target))
            tgt_len = len(self.tokenizer.encode(" " + target))
            encodings = self.tokenizer(prompt + " " + target, return_tensors="pt")
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

    def run_eval(self, model, step_idx_for_eval):
        #print("inside run_eval")
        #quit()
        model.eval()
        item_original = self.data[step_idx_for_eval]
        #print("item_original: ", item_original)
        requested_rewrite = item_original["requested_rewrite"]
        rewrite_prompt = requested_rewrite["prompt"].format(requested_rewrite["subject"])

        target_new = requested_rewrite["target_new"]["str"]
        target_true = requested_rewrite["target_true"]["str"]
        targets = [target_new, target_true]

        ppl_rewrite_prompt_target_new, ppl_rewrite_prompt_target_true = self.evaluate_model(model, rewrite_prompt, targets)


        prob_rewrite_prompt_target_new, prob_rewrite_prompt_target_true = 1 / ppl_rewrite_prompt_target_new, 1 / ppl_rewrite_prompt_target_true

        if prob_rewrite_prompt_target_new > prob_rewrite_prompt_target_true:
            self.rewrite_success_count += 1
        self.rewrite_total_count += 1
        self.rewrite_magnitude += prob_rewrite_prompt_target_new - prob_rewrite_prompt_target_true
        #print("rewrite_success_count: ", self.rewrite_success_count)



        for paraphrase in item_original["paraphrase_prompts"]:
            ppl_paraphrase_target_new, ppl_paraphrase_target_true = self.evaluate_model(model, paraphrase, targets)
            prob_paraphrase_target_new, prob_paraphrase_target_true = 1 / ppl_paraphrase_target_new, 1 / ppl_paraphrase_target_true
            
            if prob_paraphrase_target_new > prob_paraphrase_target_true:
                self.paraphrase_success_count += 1
            
            self.paraphrase_magnitude += prob_paraphrase_target_new - prob_paraphrase_target_true

            self.paraphrase_total_count += 1

        for neighbor in item_original["neighborhood_prompts"]:
            ppl_neighborhood_target_new, ppl_neighborhood_target_true = self.evaluate_model(model, neighbor, targets)
            prob_neighborhood_target_new, prob_neighborhood_target_true = 1 / ppl_neighborhood_target_new, 1 / ppl_neighborhood_target_true

            if prob_neighborhood_target_new < prob_neighborhood_target_true:
                self.neighborhood_success_count += 1

            self.neighborhood_magnitude += prob_neighborhood_target_true - prob_neighborhood_target_new

            self.neighborhood_total_count += 1

    def final_evaluation_scores(self):
        print("rewrite_success_count: ", self.rewrite_success_count)
        print("paraphrase_success_count: ", self.paraphrase_success_count)
        print("neighborhood_success_count: ", self.neighborhood_success_count)
        #quit()

        
        print("rewrite_success_accuracy: ", self.rewrite_success_count / self.rewrite_total_count)
        print("paraphrase_success_accuracy: ", self.paraphrase_success_count / self.paraphrase_total_count)
        print("neighborhood_success_accuracy: ", self.neighborhood_success_count / self.neighborhood_total_count)

        print("rewrite_magnitude: ", self.rewrite_magnitude / self.rewrite_total_count)
        print("paraphrase_magnitude: ", self.paraphrase_magnitude / self.paraphrase_total_count)
        print("neighborhood_magnitude: ", self.neighborhood_magnitude / self.neighborhood_total_count)
