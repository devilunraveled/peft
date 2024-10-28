import pandas as pd
from config import ModelConfig, Paths, FineTunedModelConfig
from src.model import Model
from utils import CustomDataset
from config import DataConfig

if __name__ == '__main__':
    testData = pd.read_csv(f"{Paths.data}/test.csv")[:3000]

    model = Model.load_fine_tuned(originalModel = ModelConfig.name, checkpoint = FineTunedModelConfig.checkpoint, weights_only = True).to("cuda")

    tokenizer = model.tokenizer
    
    print("Model Loaded")

    testDataset = CustomDataset(testData, tokenizer)
    
    print("Dataset Created")

    for i in range(1):
        input = testDataset._inference_input_(i)
        print(f"Article : {input}")
        tokenizedInput = tokenizer(input, return_tensors="pt", padding='max_length', truncation=True, max_length=DataConfig.max_length)
        inputIDs = tokenizedInput['input_ids'].to("cuda")
        attentionMask = tokenizedInput['attention_mask'].to("cuda")
        output = model.model.generate(input_ids=inputIDs, max_new_tokens=512, num_beams=5, early_stopping=True, attention_mask=attentionMask)
        print("--------------------------------------------------"*2)
        print(tokenizer.decode(output[0], skip_special_tokens=False))
        print("--------------------------------------------------"*2)
