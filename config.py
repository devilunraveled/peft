class ModelConfig :
    name = 'EleutherAI/gpt-neo-1.3b'
    
class DataConfig :
    max_length = 512

class FineTunedModelConfig :
    checkpoint = './ckpt/checkpoint-10500/'

class Paths : 
    data = "./data/cnn_dailymail/"
    model = "./ckpt/"
    logs = "./logs/"
