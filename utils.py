from torch.utils.data import Dataset
from config import DataConfig

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def _inference_input_(self, index):
        row = self.dataframe.iloc[index]
        input = row['article'] + ' ' + ". Summary :->\n" 
        return input

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input = self._inference_input_(index) + row['highlights'] 
        
        tokenizedInputs = self.tokenizer(input, return_tensors="pt", padding='max_length', truncation=True, max_length=DataConfig.max_length)

        tokenizedInputs['input_ids'] = tokenizedInputs['input_ids'].squeeze(0)
        tokenizedInputs['attention_mask'] = tokenizedInputs['attention_mask'].squeeze(0)
        tokenizedInputs['labels'] = tokenizedInputs['input_ids'].clone()

        return tokenizedInputs
