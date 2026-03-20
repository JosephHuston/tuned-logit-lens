from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# Device setup
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)

def plot_logit_evolution(prompt: str, top_k: int = 10, save_path: str = "plots/logit_lens.svg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)

    last_pos = -1
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]
    
    top_tokens = np.empty((n_layers, seq_len), dtype=object)
    top_probs  = np.zeros((n_layers, seq_len))

    for layer in range(n_layers):
        resid = cache["resid_post", layer]
        layer_logits = model.unembed(model.ln_final(resid))
        layer_probs = layer_logits.softmax(dim=-1)[0]  # (seq_len, vocab_size)
        
        top_p, top_id = layer_probs.topk(1, dim=-1)
        for pos in range(seq_len):
            token_str = model.to_string([top_id[pos, 0].item()]).strip()
            if not token_str:
                token_str = f"ID:{top_id[pos, 0].item()}"
            top_tokens[layer, pos] = token_str
            top_probs[layer, pos]  = top_p[pos, 0].item()

    token_strs = [model.to_string([t.item()]).strip() for t in tokens[0]]

    fig, ax = plt.subplots(figsize=(seq_len * 1.2, n_layers * 0.6))

    sns.heatmap(
        top_probs,
        annot=top_tokens,
        fmt="",
        cmap="viridis",
        xticklabels=token_strs,
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cbar_kws={"label": "Probability"},
        ax=ax
    )

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Input Tokens", labelpad=10)
    ax.set_ylabel("Layer")
    ax.set_title(f"Logit Lens: Top Predicted Token per Layer\nPrompt: \"{prompt}\"", pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    prompt = "The sky is blue the trees are"
    plot_logit_evolution(prompt)
