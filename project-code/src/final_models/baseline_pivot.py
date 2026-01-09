import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from .base import BaseNeutralizer
from src.model import MBartNeutralizer
from transformers import MarianMTModel, MarianTokenizer

class PivotBaseline(BaseNeutralizer):
    def __init__(self, en_neutralizer_path="models/mbart_neutralizer_en_v1"):
        super().__init__()
        
        # 1. Load ZH -> EN Translator (Helsinki-NLP)
        print("Loading ZH->EN Translator...")
        self.zh_en_name = "Helsinki-NLP/opus-mt-zh-en"
        self.tok_zh_en = MarianTokenizer.from_pretrained(self.zh_en_name)
        self.mod_zh_en = MarianMTModel.from_pretrained(self.zh_en_name).to(self.device).half() # fp16 for speed

        # 2. Load EN -> ZH Translator
        print("Loading EN->ZH Translator...")
        self.en_zh_name = "Helsinki-NLP/opus-mt-en-zh"
        self.tok_en_zh = MarianTokenizer.from_pretrained(self.en_zh_name)
        self.mod_en_zh = MarianMTModel.from_pretrained(self.en_zh_name).to(self.device).half()

        # 3. Load The Neutralizer (Your Phase 1 Model)
        print("Loading English Neutralizer...")
        self.neutralizer_wrapper = MBartNeutralizer(model_name=en_neutralizer_path)
        self.neutralizer_model = self.neutralizer_wrapper.get_model().to(self.device)
        self.neutralizer_tok = self.neutralizer_wrapper.get_tokenizer()

    def _translate(self, text, tokenizer, model):
        """Helper for translation steps"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        gen = model.generate(**inputs, num_beams=1, max_new_tokens=128) # Greedy for speed
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    def debias(self, text: str) -> str:
        # Step 1: Translate ZH -> EN
        en_biased = self._translate(text, self.tok_zh_en, self.mod_zh_en)
        
        # Step 2: Neutralize EN -> EN
        # (We use your trained English mBART here)
        self.neutralizer_tok.src_lang = "en_XX"
        encoded = self.neutralizer_tok(en_biased, return_tensors="pt").to(self.device)
        gen_ids = self.neutralizer_model.generate(
            **encoded,
            forced_bos_token_id=self.neutralizer_tok.lang_code_to_id["en_XX"],
            num_beams=3, # Slight beam search for quality
            max_length=128
        )
        en_neutral = self.neutralizer_tok.decode(gen_ids[0], skip_special_tokens=True)

        # Step 3: Translate EN -> ZH
        zh_neutral = self._translate(en_neutral, self.tok_en_zh, self.mod_en_zh)
        
        return zh_neutral