import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert HF GPT-2 small to project-native checkpoint")
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument("--out", default="Datasets/gpt2_small_converted.pt", help="Output .pt file path")
    parser.add_argument(
        "--tokenizer_dir",
        default="Datasets/gpt2_tokenizer_assets",
        help="Directory to save vocab.json and merges.txt",
    )
    args = parser.parse_args()

    # Conversion-time dependency only.
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    state = model.state_dict()
    n_layer = model.config.n_layer

    converted = {
        "config": {
            "vocab_size": model.config.vocab_size,
            "ctx_len": model.config.n_positions,
            "d_model": model.config.n_embd,
            "n_layers": model.config.n_layer,
            "n_heads": model.config.n_head,
        },
        "wte": state["transformer.wte.weight"].detach().cpu(),             # (V, C)
        "wpe": state["transformer.wpe.weight"].detach().cpu(),             # (T, C)
        "ln_f.weight": state["transformer.ln_f.weight"].detach().cpu(),    # (C,)
        "ln_f.bias": state["transformer.ln_f.bias"].detach().cpu(),        # (C,)
        "lm_head.weight": state["lm_head.weight"].detach().cpu(),          # (V, C) for LinearLayer.W
        "blocks": [],
    }

    for i in range(n_layer):
        prefix = f"transformer.h.{i}."
        converted["blocks"].append(
            {
                "ln_1.weight": state[prefix + "ln_1.weight"].detach().cpu(),
                "ln_1.bias": state[prefix + "ln_1.bias"].detach().cpu(),
                "attn.c_attn.weight": state[prefix + "attn.c_attn.weight"].detach().cpu(),
                "attn.c_attn.bias": state[prefix + "attn.c_attn.bias"].detach().cpu(),
                "attn.c_proj.weight": state[prefix + "attn.c_proj.weight"].detach().cpu(),
                "attn.c_proj.bias": state[prefix + "attn.c_proj.bias"].detach().cpu(),
                "ln_2.weight": state[prefix + "ln_2.weight"].detach().cpu(),
                "ln_2.bias": state[prefix + "ln_2.bias"].detach().cpu(),
                "mlp.c_fc.weight": state[prefix + "mlp.c_fc.weight"].detach().cpu(),
                "mlp.c_fc.bias": state[prefix + "mlp.c_fc.bias"].detach().cpu(),
                "mlp.c_proj.weight": state[prefix + "mlp.c_proj.weight"].detach().cpu(),
                "mlp.c_proj.bias": state[prefix + "mlp.c_proj.bias"].detach().cpu(),
            }
        )

    torch.save(converted, args.out)
    print(f"Saved converted checkpoint to {args.out}")
    tokenizer.save_pretrained(args.tokenizer_dir)
    print(f"Saved tokenizer assets to {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
