import pytest
import torch

from mini_gpt.model import (
    MLP,
    CausalSelfAttention,
    GPTConfig,
    MiniGPT,
    TransformerBlock,
)


@pytest.fixture
def default_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )


def test_gpt_config_defaults():
    config = GPTConfig()

    assert config.vocab_size == 50257
    assert config.block_size == 256
    assert config.n_layer == 6
    assert config.n_head == 6
    assert config.n_embd == 384


def test_causal_self_attention_shape(default_config):
    attn = CausalSelfAttention(default_config)
    x = torch.randn(2, 16, default_config.n_embd)
    out = attn(x)

    assert out.shape == (2, 16, default_config.n_embd)


def test_mlp_shape(default_config):
    mlp = MLP(default_config)
    x = torch.randn(2, 16, default_config.n_embd)
    out = mlp(x)

    assert out.shape == (2, 16, default_config.n_embd)


def test_transformer_block_shape(default_config):
    block = TransformerBlock(default_config)
    x = torch.randn(2, 16, default_config.n_embd)
    out = block(x)

    assert out.shape == (2, 16, default_config.n_embd)


def test_minigpt_inference_mode(default_config):
    model = MiniGPT(default_config)
    model.eval()
    x = torch.randint(0, default_config.vocab_size, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 1, default_config.vocab_size)
    assert loss is None


def test_minigpt_training_mode_loss(default_config):
    model = MiniGPT(default_config)
    x = torch.randint(0, default_config.vocab_size, (2, 16))
    y = torch.randint(0, default_config.vocab_size, (2, 16))
    logits, loss = model(x, y)

    assert logits.shape == (2, 16, default_config.vocab_size)
    assert loss is not None
    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0


def test_weight_tying(default_config):
    model = MiniGPT(default_config)
    assert torch.equal(model.transformer.wte.weight, model.lm_head.weight)


def test_exceed_block_size_raises_assertion(default_config):
    model = MiniGPT(default_config)
    x = torch.randint(0, default_config.vocab_size, (1, 64))

    with pytest.raises(AssertionError):
        model(x)
