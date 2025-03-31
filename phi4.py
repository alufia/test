import logging
import re
import io
import requests
import librosa
from PIL import Image
from copy import deepcopy
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig
)

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


def process_prompts(prompt: PromptStruct):
    """prompt 구조를 Qwen2audio 예시처럼 변환."""
    def _conv_contents(content):
        c = deepcopy(content)
        for ele in c:
            if ele["type"] == "audio":
                ele["audio_url"] = ele["value"]
            elif ele["type"] == "image":
                ele["image_url"] = ele["value"]
            elif ele["type"] == "text":
                ele["text"] = ele["value"]
            del ele["value"]
        return c

    for line in prompt:
        line["content"] = _conv_contents(line["contents"])
        del line["contents"]

    return prompt


class Phi4Multimodal(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        # 필요에 따라 pretraining 여부/기본 파라미터 설정
        super().__init__(True, sample_params)

        logger.debug(f"Loading Phi4 model from {path}...")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            # 필요에 따라 _attn_implementation 옵션 조정
            _attn_implementation='flash_attention_2',
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(path)
        logger.debug("Model load complete.")

    def _inference(self, prompt: PromptStruct, **kwargs):
        # 1) prompt 구조 변환
        prompt = process_prompts(prompt)

        # 2) 텍스트 변환
        #    Qwen2audio 예시처럼 apply_chat_template 사용
        #    (phi4 processor가 동일한 함수를 제공하지 않을 수 있으니 필요 시 수정)
        text = self.processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        # 3) 오디오·이미지 로드
        audios = []
        images = []
        for msg in prompt:
            for ele in msg["content"]:
                if ele["type"] == "audio":
                    audio_data, sr = librosa.load(ele["audio_url"], sr=None)
                    audios.append((audio_data, sr))
                elif ele["type"] == "image":
                    resp = requests.get(ele["image_url"])
                    image = Image.open(io.BytesIO(resp.content))
                    images.append(image)

        # 4) 입력 데이터 구성
        #    images나 audios 중 필요한 것만 processor에 넘김
        inputs = self.processor(
            text=text,
            audios=audios if audios else None,
            images=images if images else None,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        # 5) 모델 추론
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            generation_config=self.generation_config,
        )

        # 6) 결과 디코딩
        #    입력 길이만큼 잘라낸 후 batch_decode
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response
