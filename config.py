class ModelConfig :
    name = 'openai-community/gpt2'
    
class DataConfig :
    max_length = 768

class FineTunedModelConfig :
    checkpoint = './ckpt_lora/checkpoint-2624'

class Paths : 
    data = "./data/cnn_dailymail/"
    model = "./ckpt/"
    logs = "./logs/"
