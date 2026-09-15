from typing import List, Optional, Dict
from dataclasses import asdict
import numpy as np


class AudioLLMEngine:
    """
    Audio LLM inference engine wrapping vLLM for Audio Language Models.
    """
    
    def __init__(
        self,
        model_id: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
    ):
        """
        Initialize the Audio LLM Engine.

        Args:
            model_id (str): The Hugging Face model ID.
            max_model_len (int): Maximum model length.
            gpu_memory_utilization (float): GPU memory utilization.
            seed (int): Random seed.
            limit_mm_per_prompt (Optional[Dict[str, int]]): Limits for multi-modal inputs per prompt.
            trust_remote_code (bool): Whether to trust remote code for custom models.
            tensor_parallel_size (int): Number of GPUs for tensor parallelism.
            enforce_eager (bool): Disable CUDA graph compilation (helps with TP on some GPUs).
            dtype (str): Model dtype for vLLM (e.g. "bfloat16", "float16").
        """
        # Lazy imports to prevent early CUDA initialization
        from transformers import AutoProcessor
        from vllm import LLM, EngineArgs, SamplingParams

        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"audio": 1}

        engine_args = EngineArgs(
            model=model_id,
            max_model_len=max_model_len,
            limit_mm_per_prompt=limit_mm_per_prompt,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            dtype=dtype,
        )
        
        self.llm = LLM(**asdict(engine_args))
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.seed = seed
        self.sampling_params_cls = SamplingParams

    def generate(
        self,
        prompts: List[str],
        audios: Optional[List[List[np.ndarray]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256
    ) -> List[str]:
        """
        Generate responses for a batch of prompts and optional audios.

        Args:
            prompts (List[str]): List of text prompts.
            audios (Optional[List[List[np.ndarray]]]): List of audio arrays per prompt.
                   Each element is a list of audio arrays for that prompt.
                   If None, text-only inference is performed.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum new tokens to generate.

        Returns:
            List[str]: List of generated text responses.
        """
        batch_inputs = []
        use_audio = audios is not None
        
        if use_audio and len(audios) != len(prompts):
            raise ValueError("Number of audio samples must match number of prompts.")

        for i, text_prompt in enumerate(prompts):
            if use_audio:
                # Generic/Qwen-style chat template for audio
                content = [
                    {"type": "audio", "audio_url": None},
                    {"type": "text", "text": text_prompt},
                ]

                try:
                    full_prompt = self.processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                except ValueError as e:
                    if "add_generation_prompt" in str(e):
                        full_prompt = self.processor.apply_chat_template(
                            [{"role": "user", "content": content}],
                            tokenize=False
                        )
                    else:
                        raise e

                batch_inputs.append({
                    "prompt": full_prompt,
                    "multi_modal_data": {"audio": audios[i]}
                })

            else:
                content = [{"type": "text", "text": text_prompt}]

                try:
                    full_prompt = self.processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                except ValueError as e:
                    if "add_generation_prompt" in str(e):
                        full_prompt = self.processor.apply_chat_template(
                            [{"role": "user", "content": content}],
                            tokenize=False
                        )
                    else:
                        raise e
                batch_inputs.append({"prompt": full_prompt})

        sampling_params = self.sampling_params_cls(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.seed,
        )

        outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)

        return [output.outputs[0].text.strip() for output in outputs]
