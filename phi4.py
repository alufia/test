from audio_evals.models.offline_model import OfflineModel
from audio_evals.base import PromptStruct
from typing import Dict

# Hf API
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # 필요하다면 추가
    # AutoProcessor, 
    # BlipProcessor, 
    # etc.
)

class Phi4Multimodal(OfflineModel):
    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        super().__init__(is_chat, sample_params)
        # TODO: 초기화 로직
        # remote code 사용이 필요한 경우: trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True
        )
        # 멀티모달 입력(이미지/오디오 등)을 처리하려면
        # 별도의 Processor/Feature Extractor가 필요할 수 있음

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # prompt.content 등에 따라 전처리
        inputs = self.tokenizer(
            prompt.content,
            return_tensors="pt"
            # 필요에 따라 옵션 추가
        )
        # 멀티모달 입력이 있을 경우, Processor를 통해 전처리 후 모델에 넣어야 함
        # 예) processor(images=..., text=...)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)