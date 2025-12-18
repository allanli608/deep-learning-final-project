from .base import BaseNeutralizer
from src.model import MBartNeutralizer

class ZeroShotNeutralizer(BaseNeutralizer):
    def __init__(self, model_path="models/mbart_neutralizer_en_v1"):
        super().__init__()
        # Load the English-trained weights
        self.wrapper = MBartNeutralizer(model_name=model_path)
        self.model = self.wrapper.get_model().to(self.device)
        self.tokenizer = self.wrapper.get_tokenizer()
        
        # PRE-CONFIGURE FOR CHINESE
        self.tokenizer.src_lang = "zh_CN"
        self.target_lang_id = self.tokenizer.lang_code_to_id["zh_CN"]

    def debias(self, text: str) -> str:
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **encoded,
            forced_bos_token_id=self.target_lang_id, # Force Chinese Output
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)