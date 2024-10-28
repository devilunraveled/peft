# from accelerate import Accelerator
import pandas as pd
from config import Paths, ModelConfig
from src.model import Model
from transformers import Trainer, TrainingArguments
# from transformers import AdamW, get_linear_schedule_with_warmup
from utils import CustomDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def train(model, trainData, evalData, trainingArguments) :
    # accelerator = Accelerator()
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = trainingArguments['per_device_train_batch_size']
    # optimizer = AdamW(model.parameters(), lr=trainingArguments['learning_rate'])
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                             num_warmup_steps=trainingArguments['warmup_steps'],
    #                                             num_training_steps=len(trainData) * trainingArguments['num_train_epochs'])
    # model, trainData, evalData = accelerator.prepare(model, trainData, evalData)

    trainer = Trainer(model         = model, 
                      train_dataset = trainData, 
                      eval_dataset  = evalData, 
                      args          = TrainingArguments(**trainingArguments))
    trainer.train()

if __name__ == "__main__" :
    import sys
    trainData = pd.read_csv(f"{Paths.data}/train.csv")[:21000]
    evalData = pd.read_csv(f"{Paths.data}/validation.csv")[:6000]

    model = Model(modelName = ModelConfig.name, modificationType = sys.argv[1]).to("cuda")
    tokenizer = model.tokenizer
    
    print("Model Loaded")

    trainDataset = CustomDataset(trainData, tokenizer)
    evalDataset = CustomDataset(evalData, tokenizer)
    
    print("Dataset Created")

    training_arguments = {
        'output_dir': Paths.model,
        'num_train_epochs': 4,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 1,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'logging_dir': Paths.logs,
        'logging_steps': 5,
        'learning_rate': 1e-7,
        # 'bf16': True,
        'warmup_steps': 15,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'save_safetensors' : False
    }

    train(model, trainDataset, evalDataset, trainingArguments = training_arguments)
