"""The runtime's chat formatting must match Hugging Face's apply_chat_template
exactly: the same text, and so the same token ids.

The template itself is a fixture copied from the Chat model's
tokenizer_config.json at the store's pinned revision, so these tests run
offline. One test also checks token ids against the real Chat store's
tokenizer; it runs only where that store exists (models/ isn't in git).
"""

from __future__ import annotations

import json

import pytest

from expertrelay import REPO_ROOT
from expertrelay.store.chat_template import ChatTemplate, is_chat_store, render

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "qwen1.5-moe-a2.7b-chat_tokenizer_config.json"
CHAT_STORE = REPO_ROOT / "models" / "qwen1.5-moe-a2.7b-chat-int8"
# What bench/phase3_traces.py fed the Chat store for the recorded Chat traces
# (a literal string until it switched to store.chat_template).
RECORDED_CHAT_TRACE_FORMAT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
)
CONVERSATIONS = [
    [{"role": "user", "content": "The printing press changed Europe because"}],
    [{"role": "user", "content": "تعد مدينة القاهرة من أكبر المدن في العالم العربي"}],
    [{"role": "user", "content": "def f(x):\n    return {x: [1, 2]}  # braces {{ }} and {% raw %}"}],
    [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "2 + 2?"}],
    [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ],
]


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def hf_tokenizer(config: dict):
    """A real transformers tokenizer carrying the template; its vocabulary is
    irrelevant because only the rendered text is compared."""
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        chat_template=config["chat_template"],
        eos_token=config["eos_token"],
        pad_token=config["pad_token"],
    )


@pytest.mark.parametrize("messages", CONVERSATIONS)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_rendering_matches_transformers(config, messages, add_generation_prompt):
    ours = render(
        config["chat_template"],
        messages,
        add_generation_prompt=add_generation_prompt,
        special_tokens=ChatTemplate(config).special_tokens,
    )
    theirs = hf_tokenizer(config).apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    assert ours == theirs


def test_user_prompt_is_what_the_recorded_chat_traces_used(config):
    template = ChatTemplate(config)
    for text in ("Hello", "سلام عليكم", "line one\nline two"):
        assert template.user_prompt(text) == RECORDED_CHAT_TRACE_FORMAT.format(text=text)


def test_a_config_without_a_template_is_rejected():
    with pytest.raises(ValueError):
        ChatTemplate({"eos_token": "</s>"})


def test_special_tokens_saved_as_objects_are_unwrapped(config):
    cfg = {**config, "eos_token": {"content": "<|im_end|>", "special": True}}
    assert ChatTemplate(cfg).special_tokens["eos_token"] == "<|im_end|>"


def test_only_chat_stores_get_the_template_by_default(tmp_path):
    for repo, expected in (("Qwen/Qwen1.5-MoE-A2.7B-Chat", True), ("Qwen/Qwen1.5-MoE-A2.7B", False)):
        (tmp_path / "store.json").write_text(json.dumps({"source": {"repo_id": repo}}))
        assert is_chat_store(tmp_path) is expected


@pytest.mark.skipif(
    not (CHAT_STORE / "tokenizer_config.json").exists() or not (CHAT_STORE / "tokenizer.json").exists(),
    reason="the real Chat store (with its tokenizer_config.json) isn't on this machine",
)
def test_token_ids_match_transformers_on_the_real_chat_tokenizer():
    from expertrelay.store.tokenizer import load_tokenizer

    hf = transformers.AutoTokenizer.from_pretrained(str(CHAT_STORE))
    ours = load_tokenizer(CHAT_STORE)
    template = ChatTemplate.for_store(CHAT_STORE)
    for messages in CONVERSATIONS[:3]:
        expected = hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        expected = (
            expected["input_ids"] if isinstance(expected, dict) or hasattr(expected, "keys") else expected
        )
        assert ours.encode(template.user_prompt(messages[0]["content"])).ids == list(expected)
