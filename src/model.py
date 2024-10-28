import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils import configure_lora
from peft import prepare_model_for_kbit_training

class Model(torch.nn.Module) :
    def __init__(self, modelName, modificationType = None, *args, **kwargs) :
        super().__init__()
        quantizationConfig = BitsAndBytesConfig()

        self._model = AutoModelForCausalLM.from_pretrained(modelName)
        self._tokenizer = AutoTokenizer.from_pretrained(modelName)
        
        if modificationType :
            self._modify_model(modificationType, *args, **kwargs)
        self._modify_tokenizer()
    
    @classmethod
    def load_fine_tuned(cls, originalModel, checkpoint, *args, **kwargs) :
        self = cls.__new__(cls)
        super().__init__(self)
        # Instantiate the parent class.
        self._model = AutoModelForCausalLM.from_pretrained(originalModel)
        self._tokenizer = AutoTokenizer.from_pretrained(originalModel)
        self._modify_tokenizer()
        self.load_state_dict(torch.load(f"{checkpoint}/pytorch_model.bin", *args, **kwargs))
        return self
    
    @property
    def model(self) :
        return self._model

    @property
    def tokenizer(self) :
        return self._tokenizer

    def _modify_tokenizer(self):
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.add_special_tokens({'additional_special_tokens': ["<|summary|>"]})
        print(self._tokenizer.additional_special_tokens)
        print(f"Pad Token : {self.tokenizer.pad_token}")

    def _modify_model(self, modificationType, *args, **kwargs) :
        self._model = prepare_model_for_kbit_training(self._model)
        if modificationType == 'lora' :
            self._model = configure_lora(self._model, *args, **kwargs)
        elif modificationType == 'unfreeze_lm_head' :
            for param in self._model.parameters() :
                param.requires_grad = False

            for param in self._model.lm_head.parameters() :
                param.requires_grad = True
        else :
            raise NotImplementedError

    def forward(self, *args, **kwargs) :
        return self.model(*args, **kwargs)
