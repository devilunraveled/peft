import pandas as pd
from config import ModelConfig, Paths, FineTunedModelConfig
from src.model import Model
from utils import CustomDataset
from config import DataConfig
from rouge_score import rouge_scorer

if __name__ == '__main__':
    import sys
    testData = pd.read_csv(f"{Paths.data}/test.csv")[:3000]

    model = Model.load_fine_tuned(originalModel = ModelConfig.name, checkpoint = FineTunedModelConfig.checkpoint, weights_only = True, modificationType = sys.argv[1])

    tokenizer = model.tokenizer
    
    print("Model Loaded")

    testDataset = CustomDataset(testData, tokenizer)
    
    print("Dataset Created")
    obtained = []
    scores = []
    target = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in range(5):
        # print(f"Training Sample : \n{tokenizer.decode(testDataset[i]['input_ids'])}")
        input = testDataset._inference_input_(i)
        print(f"Article : {input}")
        tokenizedInput = tokenizer(input, return_tensors="pt", padding='max_length', truncation=True, max_length=DataConfig.max_length)
        inputIDs = tokenizedInput['input_ids']
        attentionMask = tokenizedInput['attention_mask']
        output = model.model.generate(input_ids=inputIDs, max_new_tokens=512, attention_mask=attentionMask, num_beams=5, no_repeat_ngram_size=3)
        print("--------------------------------------------------"*2)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("--------------------------------------------------"*2)
        
        obtained.append(tokenizer.decode(output[0], skip_special_tokens=False)[len(input):])
        target.append(testData.iloc[i]['highlights'])

        scores.append(scorer.score(obtained[-1], target[-1]))

    print(scores)
