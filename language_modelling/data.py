from torch.utils.data import Dataset

class FinetuneDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(self.data[idx], padding="longest", return_tensors="pt")
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs, self.data[idx]

