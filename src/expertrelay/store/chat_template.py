"""The model's own chat template, kept with the store and rendered the way
Hugging Face transformers renders it.

A chat model is trained on prompts in its template; feeding it raw text
gives outputs (and expert routing) it was never tuned for. The template is
the `chat_template` string in the model's tokenizer_config.json, fetched at
the store's pinned revision and kept in the store directory next to
tokenizer.json, so the same commit hash pins it too.

Rendering mirrors transformers' `apply_chat_template` (transformers 5.x,
utils.chat_template_utils): a Jinja2 ImmutableSandboxedEnvironment with
trim_blocks, lstrip_blocks and the loopcontrols extension, a `tojson`
filter, `raise_exception` and `strftime_now` globals, and the tokenizer's
special tokens passed as template variables.
tests/test_chat_template.py checks the result against transformers itself.
Tokenize the rendered text with the store's tokenizer as-is: the template
already contains every special token, and Qwen adds no BOS.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from expertrelay.store.fetch_hf_tensors import fetch_file

TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_SPECIAL_TOKENS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def ensure_tokenizer_config(store_dir: Path) -> Path:
    path = Path(store_dir) / TOKENIZER_CONFIG_FILE
    if not path.exists():
        source = json.loads((Path(store_dir) / "store.json").read_text())["source"]
        path.write_bytes(fetch_file(source["repo_id"], TOKENIZER_CONFIG_FILE, source["revision"]))
    return path


def is_chat_store(store_dir: Path) -> bool:
    """Whether prompts get the chat template by default: the store was built
    from a chat model. Base models may ship a template too, but are run on
    raw text."""
    return json.loads((Path(store_dir) / "store.json").read_text())["source"]["repo_id"].endswith("-Chat")


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def render(
    template: str,
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
    special_tokens: dict | None = None,
) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(template).render(
        messages=messages,
        tools=None,
        documents=None,
        add_generation_prompt=add_generation_prompt,
        **(special_tokens or {}),
    )


class ChatTemplate:
    def __init__(self, tokenizer_config: dict):
        template = tokenizer_config.get("chat_template")
        if not isinstance(template, str):
            raise ValueError("tokenizer_config.json has no single chat_template string")
        self.template = template
        self.special_tokens = {}
        for k in _SPECIAL_TOKENS:
            v = tokenizer_config.get(k)
            if isinstance(v, dict):  # an AddedToken serialized as {"content": ...}
                v = v.get("content")
            if v is not None:
                self.special_tokens[k] = v

    @classmethod
    def for_store(cls, store_dir: Path) -> ChatTemplate:
        return cls(json.loads(ensure_tokenizer_config(store_dir).read_text(encoding="utf-8")))

    def user_prompt(self, text: str) -> str:
        """One user turn, the template's default system prompt, and the
        assistant header the model continues from."""
        return render(
            self.template,
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            special_tokens=self.special_tokens,
        )
