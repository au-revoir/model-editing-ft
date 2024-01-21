import torch

#PROMPT_TEMPLATE = "<s>[INST] Generate 30 paraphrases. Write the paraphrases in the same format as: {} {}[/INST]"
#PROMPT_TEMPLATE = "<s>[INST] Statement: South East Cape, located in? Answer: Australia \nParaphrases: \n1 South East Cape is located in? Answer: Australia \n2 South East Cape, located in? Answer: Australia \nStatement: Renato Augusto, the? Answer: midfielder \nParaphrases: \n1 Renato Augusto, who plays the position? Answer: midfielder \n2 Which position does Renato Augusto play? They play as? Answer: midfielder \nStatement: Edinburgh Mathematical Society works in the area of? Answer: mathematics \nParaphrases: \n1 Edinburgh Mathematical Society works in the field of? Answer: mathematics \n2 Edinburgh Mathematical Society's domain of work is? Answer: mathematics \nStatement: {}? Answer: {} Generate 5 paraphrases. [/INST]"

PROMPT_TEMPLATE = "<s>[INST] Question: South East Cape, located in? Answer: Australia \nQuestion Paraphrases: \n1 South East Cape is located in? Answer: Australia \n2 South East Cape, located in? Answer: Australia \nQuestion: Renato Augusto, the? Answer: midfielder \nQuestion Paraphrases: \n1 Renato Augusto, who plays the position? Answer: midfielder \n2 Which position does Renato Augusto play? They play as? Answer: midfielder \nQuestion: Edinburgh Mathematical Society works in the area of? Answer: mathematics \nQuestion Paraphrases: \n1 Edinburgh Mathematical Society works in the field of? Answer: mathematics \n2 Edinburgh Mathematical Society's domain of work is? Answer: mathematics \nQuestion: {}? Answer: {} Generate 10 paraphrases using the format <Question Paraphrase:, Answer: {}> [/INST]"
#PROMPT_TEMPLATE = "<s>[INST] Sentence: South East Cape, located in? Answer: Australia \nSentence Paraphrases: \n1 South East Cape is located in? Answer: Australia \n2 South East Cape, located in? Answer: Australia \nSentence: Renato Augusto, the? Answer: midfielder \nSentence Paraphrases: \n1 Renato Augusto, who plays the position? Answer: midfielder \n2 Which position does Renato Augusto play? They play as? Answer: midfielder \nSentence: Edinburgh Mathematical Society works in the area of? Answer: mathematics \nSentence Paraphrases: \n1 Edinburgh Mathematical Society works in the field of? Answer: mathematics \n2 Edinburgh Mathematical Society's domain of work is? Answer: mathematics \nSentence: {}? Answer: {}. Generate 10 paraphrases using the format <Sentence Paraphrase:, Answer: {}> [/INST]"

#PROMPT_TEMPLATE = "Sentence: South East Cape, located in? Answer: Australia \nSentence Paraphrases: \n1 South East Cape is located in? Answer: Australia \n2 South East Cape, located in? Answer: Australia \nSentence: Renato Augusto, the? Answer: midfielder \nSentence Paraphrases: \n1 Renato Augusto, who plays the position? Answer: midfielder \n2 Which position does Renato Augusto play? They play as? Answer: midfielder \nSentence: Edinburgh Mathematical Society works in the area of? Answer: mathematics \nSentence Paraphrases: \n1 Edinburgh Mathematical Society works in the field of? Answer: mathematics \n2 Edinburgh Mathematical Society's domain of work is? Answer: mathematics \nSentence: {}? Answer: {}. Generate 10 paraphrases using the format <Sentence Paraphrase:, Answer: {}>"

#PROMPT_TEMPLATE = "Question: South East Cape, which is located in? Subject: South East Cape. Answer: Australia. Similar Examples: 1. Darwin International Airport, which is located in? Answer: Australia 2. Maningrida Airport, in? Answer: Australia 3. Tamworth Regional Airport, located in? Answer: Australia\nQuestion: Renato Augusto, the? Subject: Renato Augusto. Answer: midfielder. Similar Examples: 1. Which position does Rainer Bonhof play? They play as? Answer: midfielder 2. Zico, the? Answer: midfielder 3. Rainer Bonhof plays in the position of? Answer: midfielder\nQuestion: Edinburgh Mathematical Society works in the area of? Subject: Edinburgh Mathematical Society. Answer: mathematics. Similar Examples: 1. The expertise of Joseph Fourier is? Answer: mathematics 2. The expertise of Ptolemy is? Answer: mathematics 3. The expertise of Leonhard Euler is? Answer: mathematics\nQuestion: medical association's expertise is? Subject: medical association. Answer: medicine. Similar Examples: 1. Ahliman Amiraslanov works in the area of? Answer: medicine 2. Arkadij Alfonskij's domain of work is? Answer: medicine 3. The domain of activity of Mukhtar Aliev is? Answer: medicine\nQuestion: Jaap Spaanderman plays the? Subject: Jaap Spaanderman. Answer: piano. Similar Examples: 1. Peter Igelhoff plays the? Answer: piano 2. Paul Badura-Skoda, the? Answer: piano 3. Peter Igelhoff plays? Answer: piano\nQuestion: {}? Subject: {}. Answer: {}. Generate 30 more Similar Examples with different Subject in the format: <Question: , Subject: , Answer: {}>"

PROMPT_TEMPLATE = "<s>[INST] Sentence: {}...; Subject: {}; Completion: {}. Generate 20 similar sentences with the same completion but different subject in the format: <Sentence: ;Subject: ; Completion: {}> [/INST]"

class CFDataset(torch.utils.data.Dataset):
    def __init__(self, data, icl_type):
        self.data = data
        self.icl_type = icl_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_true = self.data[idx]["requested_rewrite"]["target_true"]["str"]
        subject = self.data[idx]["requested_rewrite"]["subject"]

        if self.icl_type == "paraphrase":
            text = PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format(subject), target_true, target_true)
            single_turn_prompt = f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:" #for starling-lm-7b-alpha

            item = {"case_id": self.data[idx]["case_id"], 
                    "requested_rewrite": {"prompt": self.data[idx]["requested_rewrite"]["prompt"],
                    "subject": self.data[idx]["requested_rewrite"]["subject"],
                    "target_new": self.data[idx]["requested_rewrite"]["target_new"]["str"],
                    "target_true": self.data[idx]["requested_rewrite"]["target_true"]["str"],
                    "relation_id": self.data[idx]["requested_rewrite"]["relation_id"]},
                    #"text": PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format(self.data[idx]["requested_rewrite"]["subject"]), self.data[idx]["requested_rewrite"]["target_true"]["str"])
                    #"text": PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format(self.data[idx]["requested_rewrite"]["subject"]), self.data[idx]["requested_rewrite"]["target_true"]["str"])
                    "text": PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format(subject), target_true, target_true)
                    #"text": single_turn_prompt
                    }
        if self.icl_type == "neighborhood":
            #text = PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format(subject), subject, target_true, target_true)
            
            #single_turn_prompt = f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
            text = PROMPT_TEMPLATE.format(self.data[idx]["requested_rewrite"]["prompt"].format({subject}), subject, target_true, target_true)

            item = {"case_id": self.data[idx]["case_id"], 
                    "requested_rewrite": {"prompt": self.data[idx]["requested_rewrite"]["prompt"],
                    "subject": self.data[idx]["requested_rewrite"]["subject"],
                    "target_new": self.data[idx]["requested_rewrite"]["target_new"]["str"],
                    "target_true": self.data[idx]["requested_rewrite"]["target_true"]["str"],
                    "relation_id": self.data[idx]["requested_rewrite"]["relation_id"]},
                    #"text": single_turn_prompt
                    "text": text
                    }
        return item
