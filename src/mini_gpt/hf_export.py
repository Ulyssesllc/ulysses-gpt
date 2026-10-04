import argparse
from pathlib import Path

import torch
from torch import Tensor

from .model import MiniGPT


def export_to_huggingface(
    model: MiniGPT,
    output_dir: str | Path,
    model_name: str = "ulysses-gpt",
) -> Path:

    try:
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise ImportError(
            "Hugging Face export requires transformers and safetensors. "
            "Install the project dependencies with `pip install -e .`."
        ) from exc

    source_config = model.config
    if not source_config.bias:
        raise ValueError("Hugging Face GPT-2 export currently requires bias=True.")

    config = GPT2Config(
        vocab_size=source_config.vocab_size,
        n_positions=source_config.block_size,
        n_embd=source_config.n_embd,
        n_layer=source_config.n_layer,
        n_head=source_config.n_head,
        activation_function="gelu",
        resid_pdrop=source_config.dropout,
        embd_pdrop=source_config.dropout,
        attn_pdrop=source_config.dropout,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        tie_word_embeddings=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    setattr(config, "model_name", model_name)
    setattr(config, "name_or_path", model_name)
    setattr(config, "_name_or_path", model_name)
    config.architectures = ["GPT2LMHeadModel"]

    hf_model = GPT2LMHeadModel(config)  # type: ignore[no-untyped-call]
    source_state = model.state_dict()
    converted_state: dict[str, Tensor] = {}
    linear_projection_weights = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    for name, tensor in source_state.items():
        if name.endswith(".attn.bias"):
            continue
        if name.endswith(linear_projection_weights):
            converted_state[name] = tensor.detach().T.contiguous().cpu()
        else:
            converted_state[name] = tensor.detach().cpu()

    incompatible = hf_model.load_state_dict(converted_state, strict=False)
    parameter_names = {name for name, _ in hf_model.named_parameters()}
    missing_parameters = parameter_names.intersection(incompatible.missing_keys)
    if missing_parameters or incompatible.unexpected_keys:
        raise RuntimeError(
            "Could not map MiniGPT weights to Hugging Face GPT-2. "
            f"Missing parameters: {sorted(missing_parameters)}; "
            f"unexpected keys: {incompatible.unexpected_keys}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hf_model.eval()  # type: ignore[no-untyped-call]
    hf_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.model_max_length = source_config.block_size
    tokenizer.save_pretrained(output_path)

    (output_path / "README.md").write_text(
        f"# {model_name}\n\n"
        "A decoder-only GPT language model trained with the ulysses-gpt project. "
        "The model follows the Hugging Face GPT-2 architecture and can be loaded "
        "with `AutoModelForCausalLM`.\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ulysses-gpt to Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Final native .pt checkpoint"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Hugging Face model directory"
    )
    parser.add_argument("--model-name", default="ulysses-gpt")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config.")

    model = MiniGPT(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    output_path = export_to_huggingface(
        model,
        args.output_dir,
        model_name=args.model_name,
    )
    print(f"Hugging Face model saved to: {output_path}")


if __name__ == "__main__":
    main()
