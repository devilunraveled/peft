import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils import configure_lora
from peft import prepare_model_for_kbit_training

quantizationConfig = BitsAndBytesConfig(load_in_8bit = True)

class Model(torch.nn.Module) :
    def __init__(self, modelName, modificationType = None, num_soft_prompt_tokens = 5 ,*args, **kwargs) :
        super().__init__()

        self._model = AutoModelForCausalLM.from_pretrained(modelName, quantization_config = quantizationConfig)
        self._tokenizer = AutoTokenizer.from_pretrained(modelName)
        
        if modificationType :
            self.modification_type = modificationType
            self.num_soft_prompt_tokens = num_soft_prompt_tokens
            self._modify_model(self.modification_type, *args, **kwargs)
        self._modify_tokenizer()
    
    @classmethod
    def load_fine_tuned(cls, originalModel, checkpoint, modificationType = None, num_soft_prompt_tokens = 5, *args, **kwargs) :
        self = cls.__new__(cls)
        super().__init__(self)
        # Instantiate the parent class.
        self._model = AutoModelForCausalLM.from_pretrained(originalModel, quantization_config = quantizationConfig)
        self._tokenizer = AutoTokenizer.from_pretrained(originalModel)
        
        self._modify_tokenizer()

        if modificationType :
            self.modification_type = modificationType
            self._modify_model(self.modification_type)
            if modificationType == 'soft_prompt_tuning' :
                self.num_soft_prompt_tokens = num_soft_prompt_tokens

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
        self._tokenizer.add_special_tokens({'additional_special_tokens': ["<|SUMMARIZE|>"]})
        print(self._tokenizer.additional_special_tokens)
        print(f"Pad Token : {self.tokenizer.pad_token}")

    def _modify_model(self, modificationType, *args, **kwargs) :
        self._model = prepare_model_for_kbit_training(self._model)
        if modificationType == 'lora' :
            self._model = configure_lora(self._model, *args, **kwargs)
        elif modificationType == 'unfreeze_lm_head' :
            self._freeze_model_()
            for param in self._model.lm_head.parameters() :
                param.requires_grad = True
        elif modificationType == 'soft_prompt_tuning' :
            self._freeze_model_()
            self._model.soft_prompt = torch.nn.Embedding(self.num_soft_prompt_tokens, self._model.config.n_embd)
        else :
            raise NotImplementedError
    
    def _freeze_model_(self):
        for param in self._model.parameters() :
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        if self.modification_type == 'soft_prompt_tuning':
            input_ids = kwargs.pop('input_ids')
            attention_mask = kwargs.pop('attention_mask')
            batch_size = input_ids.shape[0]
            
            # Generate soft prompt tokens
            soft_prompt_tokens = self._model.soft_prompt.weight.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_soft_prompt_tokens, embedding_dim)
            
            # Get input embeddings
            input_embeddings = self._model.transformer.wte(input_ids)  # Shape: (batch_size, seq_length, embedding_dim)
            
            # Concatenate soft prompt embeddings with input embeddings
            combined_embeddings = torch.cat((soft_prompt_tokens, input_embeddings), dim=1)  # Shape: (batch_size, num_soft_prompt_tokens + seq_length, embedding_dim)

            # Create a new attention mask that accounts for the soft prompts
            combined_attention_mask = torch.ones(combined_embeddings.shape[1], device=input_ids.device)  # Shape: (num_soft_prompt_tokens + seq_length,)
            combined_attention_mask[:self.num_soft_prompt_tokens] = 1  # Soft prompts are attended to
            combined_attention_mask[self.num_soft_prompt_tokens:] = attention_mask  # Original attention mask for the rest

            combined_attention_mask = combined_attention_mask.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_soft_prompt_tokens + seq_length)

            # Forward pass through the model with combined embeddings and attention mask
            outputs = self.model(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask, **kwargs)
            
            return outputs
        else:
            return self.model(*args, **kwargs)
