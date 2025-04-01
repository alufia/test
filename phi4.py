import logging
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import librosa
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from audio_evals.base import PromptStruct
from copy import deepcopy

from audio_evals.models.model import Model  # 프레임워크에 맞게 수정 가능

logger = logging.getLogger(__name__)

def process_prompts(prompt: PromptStruct):
    def _conv_contents(content):
        content = deepcopy(content)
        for ele in content:
            if ele["type"] == "audio":
                ele["audio_url"] = ele["value"]
            elif ele["type"] == "text":
                ele["text"] = ele["value"]
            del ele["value"]
        return content

    for line in prompt:
        line["content"] = _conv_contents(line["contents"])
        del line["contents"]

    return prompt

class Phi4MultimodalInstruct(Model):
    def __init__(self, path: str, sample_params: Optional[Dict[str, Any]] = None):
        super().__init__(False, sample_params)
        logger.debug(f"모델을 {path}에서 로드하는 중...")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(path)
        logger.debug(f"{path}에서 모델 로드 완료")

    def _inference(
        self,
        prompt: PromptStruct,
        **kwargs,
    ) -> str:
        prompt = process_prompts(prompt)
        text = self.processor.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

        audios = []
        for message in prompt:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append( 
                            librosa.load(
                                ele["audio_url"],
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )
        

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."}
        ] + prompt
        logger.debug("prompt: {}".format(prompt))
        inputs = self.processor(text=text, audios=audios, return_tensors="pt").to("cuda")

        # 모델 생성
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
            **kwargs,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response
