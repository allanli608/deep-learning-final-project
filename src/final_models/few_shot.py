from .base import BaseNeutralizer
from src.model import MBartNeutralizer

class FewShotNeutralizer(BaseNeutralizer):
    def __init__(self, model_path="models/mbart_neutralizer_zh_1percent"):
        super().__init__()
        # Load the Low-Resource (1% trained) weights
        # We explicitly set langs to zh_CN because this model was fine-tuned on Chinese
        self.wrapper = MBartNeutralizer(
            model_name=model_path, 
            src_lang="zh_CN", 
            tgt_lang="zh_CN"
        )
        self.model = self.wrapper.get_model().to(self.device)
        self.tokenizer = self.wrapper.get_tokenizer()
        self.target_lang_id = self.tokenizer.lang_code_to_id["zh_CN"]

    def debias(self, text: str) -> str:
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **encoded,
            forced_bos_token_id=self.target_lang_id,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)