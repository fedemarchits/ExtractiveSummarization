
"""Generation backends behind one interface.

    backend.generate_batch(prompts, system=None) -> list[str]

Backends:
- local:
    Hugging Face Transformers on GPU for small and medium models.
- vllm:
    vLLM local inference with continuous batching.
- endpoint:
    OpenAI-compatible chat API configured through:
        OPENAI_BASE_URL
        OPENAI_API_KEY

get_backend(alias, models_yaml) reads models.yaml, locates the model alias,
and constructs the backend specified by its ``backend`` field.

Heavy dependencies such as torch, transformers, vllm, and openai are imported
lazily so this module can be imported for offline configuration inspection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local generation.

    This affects the local Hugging Face backend. vLLM and endpoint backends
    receive their seeds through their own generation APIs.
    """
    import numpy as np
    import torch

    normalized_seed = int(seed)

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(normalized_seed)
        torch.cuda.manual_seed_all(normalized_seed)

    # These settings improve repeatability for supported CUDA operations.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class GenConfig:
    """Shared generation settings."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42


# --------------------------------------------------------------------------
# Local Hugging Face backend
# --------------------------------------------------------------------------


class LocalHFBackend:
    """Hugging Face Transformers generation backend."""

    wants_full_batch = False

    def __init__(
        self,
        hf_id: str,
        gen: GenConfig,
        device_map=None,
        load_in_4bit: bool = False,
    ):
        self.hf_id = hf_id
        self.gen = gen

        # Default to a single GPU. ``device_map="auto"`` may be configured in
        # models.yaml for models that require sharding.
        self.device_map = (
            device_map
            if device_map is not None
            else {"": 0}
        )

        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tok = None

        # The first generation call uses gen.seed, the next uses gen.seed + 1,
        # and so forth. This produces distinct but reproducible sampled calls.
        self._generation_call_index = 0

    def _ensure(self) -> None:
        """Load the tokenizer and model once."""
        if self._model is not None:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        # Seed model initialization and any library-side random operations.
        set_random_seed(self.gen.seed)

        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16

        # Prevent device_map="auto" from spilling model layers to CPU or disk.
        if self.device_map == "auto":
            gpu_count = torch.cuda.device_count()

            if gpu_count <= 0:
                raise RuntimeError(
                    "device_map='auto' was requested, but no CUDA GPU "
                    "is available."
                )

            model_kwargs["max_memory"] = {
                gpu_index: "23GiB"
                for gpu_index in range(gpu_count)
            }

        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            **model_kwargs,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.hf_id!r} has neither a pad token "
                    "nor an EOS token."
                )

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        self._model = model.eval()
        self._tok = tokenizer

    def _next_generation_seed(self) -> int:
        """Return the next deterministic local-generation seed."""
        seed = (
            int(self.gen.seed)
            + self._generation_call_index
        )

        self._generation_call_index += 1
        return seed

    def reset_generation_seed(self) -> None:
        """Reset local generation to the configured base seed.

        This is useful for tests or when deliberately restarting the complete
        generation sequence. Do not call it between self-consistency samples.
        """
        self._generation_call_index = 0

    def _input_device(self):
        """Return a device suitable for tokenizer output tensors."""
        import torch

        model = self._model

        # For ordinary non-sharded models, model.device is sufficient.
        try:
            device = model.device

            if str(device) != "meta":
                return device
        except (AttributeError, RuntimeError):
            pass

        # Accelerate-dispatched models expose a Hugging Face device map.
        hf_device_map = getattr(model, "hf_device_map", None)

        if isinstance(hf_device_map, dict):
            for mapped_device in hf_device_map.values():
                if mapped_device in {"cpu", "disk"}:
                    continue

                if isinstance(mapped_device, int):
                    return torch.device(
                        f"cuda:{mapped_device}"
                    )

                mapped_device = str(mapped_device)

                if mapped_device.startswith("cuda"):
                    return torch.device(mapped_device)

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    def _gen_chats(
        self,
        chats: List[str],
    ) -> List[str]:
        """Generate outputs for already rendered chat-template strings."""
        import torch

        if not chats:
            return []

        tokenizer = self._tok
        model = self._model

        if tokenizer is None or model is None:
            raise RuntimeError(
                "Local Hugging Face backend was not initialized."
            )

        generation_seed = self._next_generation_seed()
        set_random_seed(generation_seed)

        encoded = tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
        )

        input_device = self._input_device()

        encoded = {
            key: tensor.to(input_device)
            for key, tensor in encoded.items()
        }

        prompt_lengths = encoded[
            "attention_mask"
        ].sum(dim=1)

        do_sample = float(self.gen.temperature) > 0.0

        generation_kwargs = {
            "max_new_tokens": int(
                self.gen.max_new_tokens
            ),
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Temperature should not be passed for deterministic greedy decoding.
        if do_sample:
            generation_kwargs["temperature"] = float(
                self.gen.temperature
            )

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )

        texts = [
            tokenizer.decode(
                output_row[int(prompt_length):],
                skip_special_tokens=True,
            )
            for output_row, prompt_length in zip(
                generated,
                prompt_lengths,
            )
        ]

        del encoded
        del generated

        return texts

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one output for each prompt."""
        import torch

        if not prompts:
            return []

        self._ensure()

        tokenizer = self._tok

        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer was not initialized."
            )

        template_kwargs = {}

        # Disable Qwen thinking mode because the benchmark requires a direct
        # JSON response rather than a long internal reasoning trace.
        if "qwen" in self.hf_id.lower():
            template_kwargs["enable_thinking"] = False

        chats: List[str] = []

        for prompt in prompts:
            messages = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

            chats.append(rendered)

        # Try the complete batch first. If the batch is too large, retry each
        # prompt individually while clearing unused CUDA cache between calls.
        try:
            return self._gen_chats(chats)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            texts: List[str] = []

            for chat in chats:
                texts.extend(
                    self._gen_chats([chat])
                )

                torch.cuda.empty_cache()

            return texts

    def free_memory(self) -> None:
        """Release unused CUDA cache without unloading the model."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# vLLM backend
# --------------------------------------------------------------------------


class VLLMBackend:
    """Fast local inference using vLLM continuous batching."""

    # The runner may send all prompts for a variant in one call.
    wants_full_batch = True

    def __init__(
        self,
        hf_id: str,
        gen: GenConfig,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        disable_reasoning: bool = True,
    ):
        self.hf_id = hf_id
        self.gen = gen
        self.quantization = quantization
        self.tp = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.disable_reasoning = disable_reasoning

        self._llm = None
        self._tok = None
        self._sp = None

    def _ensure(self) -> None:
        """Initialize vLLM and its sampling parameters."""
        if self._llm is not None:
            return

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tok = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
        )

        kwargs = {
            "model": self.hf_id,
            "trust_remote_code": True,
            "tensor_parallel_size": int(self.tp),
            "gpu_memory_utilization": float(
                self.gpu_mem
            ),
            "dtype": "bfloat16",
        }

        if self.max_model_len:
            kwargs["max_model_len"] = int(
                self.max_model_len
            )

        if self.quantization:
            kwargs["quantization"] = self.quantization

            if self.quantization == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"

        self._llm = LLM(**kwargs)

        self._sp = SamplingParams(
            temperature=float(self.gen.temperature),
            max_tokens=int(self.gen.max_new_tokens),
            seed=(
                int(self.gen.seed)
                if self.gen.temperature > 0
                else None
            ),
        )

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one vLLM output per prompt."""
        if not prompts:
            return []

        self._ensure()

        tokenizer = self._tok

        template_kwargs = {}

        if (
            "qwen" in self.hf_id.lower()
            and self.disable_reasoning
        ):
            template_kwargs["enable_thinking"] = False

        chats: List[str] = []

        for prompt in prompts:
            messages = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            chats.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )

        outputs = self._llm.generate(
            chats,
            self._sp,
            use_tqdm=False,
        )

        return [
            output.outputs[0].text
            for output in outputs
        ]

    def free_memory(self) -> None:
        """vLLM manages its own model and KV-cache memory."""
        return None


# --------------------------------------------------------------------------
# OpenAI-compatible endpoint backend
# --------------------------------------------------------------------------


class EndpointBackend:
    """OpenAI-compatible remote chat backend."""

    wants_full_batch = False

    def __init__(
        self,
        model_id: str,
        gen: GenConfig,
        disable_reasoning: bool = True,
    ):
        self.model_id = model_id
        self.gen = gen
        self.disable_reasoning = disable_reasoning
        self._client = None

    def _ensure(self) -> None:
        """Create the OpenAI-compatible client."""
        if self._client is not None:
            return

        import os

        from openai import OpenAI

        base_url = os.environ.get(
            "OPENAI_BASE_URL"
        )
        api_key = os.environ.get(
            "OPENAI_API_KEY"
        )

        if not base_url:
            raise EnvironmentError(
                "OPENAI_BASE_URL is required for endpoint models."
            )

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for endpoint models."
            )

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(
                os.environ.get(
                    "OPENROUTER_TIMEOUT",
                    "120",
                )
            ),
            max_retries=2,
        )

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one remote response per prompt."""
        if not prompts:
            return []

        self._ensure()

        outputs: List[str] = []

        for prompt in prompts:
            messages = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            extra_body = (
                {"reasoning": {"enabled": False}}
                if self.disable_reasoning
                else None
            )

            response = (
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=int(
                        self.gen.max_new_tokens
                    ),
                    temperature=float(
                        self.gen.temperature
                    ),
                    seed=int(self.gen.seed),
                    extra_body=extra_body,
                )
            )

            outputs.append(
                response.choices[0].message.content
                or ""
            )

        return outputs

    def free_memory(self) -> None:
        """Remote endpoints have no local model memory to release."""
        return None


# --------------------------------------------------------------------------
# Backend factory
# --------------------------------------------------------------------------


def _find_model(
    alias: str,
    models_yaml: str,
) -> Dict:
    """Find one model configuration by alias."""
    path = Path(models_yaml)

    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    models = config.get("models", [])

    for model in models:
        if model.get("alias") == alias:
            return model

    available = [
        model.get("alias")
        for model in models
        if model.get("alias")
    ]

    raise KeyError(
        f"Model alias {alias!r} was not found in {models_yaml}. "
        f"Available aliases: {available}"
    )


def get_backend(
    alias: str,
    models_yaml: str,
    gen: Optional[GenConfig] = None,
):
    """Construct the configured generation backend."""
    model_config = _find_model(
        alias,
        models_yaml,
    )

    generation_config = gen or GenConfig()

    backend_kind = model_config.get(
        "backend",
        "local",
    )

    if backend_kind == "local":
        return LocalHFBackend(
            hf_id=model_config["hf_id"],
            gen=generation_config,
            device_map=model_config.get(
                "device_map"
            ),
            load_in_4bit=bool(
                model_config.get(
                    "load_in_4bit",
                    False,
                )
            ),
        )

    if backend_kind == "vllm":
        return VLLMBackend(
            hf_id=model_config["hf_id"],
            gen=generation_config,
            quantization=model_config.get(
                "quantization"
            ),
            tensor_parallel_size=int(
                model_config.get(
                    "tensor_parallel_size",
                    1,
                )
            ),
            max_model_len=model_config.get(
                "max_model_len"
            ),
            gpu_memory_utilization=float(
                model_config.get(
                    "gpu_memory_utilization",
                    0.90,
                )
            ),
            disable_reasoning=bool(
                model_config.get(
                    "disable_reasoning",
                    True,
                )
            ),
        )

    if backend_kind == "endpoint":
        return EndpointBackend(
            model_id=model_config.get(
                "endpoint_model",
                model_config["hf_id"],
            ),
            gen=generation_config,
            disable_reasoning=bool(
                model_config.get(
                    "disable_reasoning",
                    True,
                )
            ),
        )

    raise ValueError(
        f"Unknown backend {backend_kind!r} "
        f"for model alias {alias!r}."
    )

