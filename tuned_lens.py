from transformer_lens import HookedTransformer
import torch
import torch.nn as nn

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained("gpt2", dtype=torch.float32)
model.eval().to(device)

class TunedLens(torch.nn.Module):
    """One affine probe per layer, each mapping residual stream -> residual stream."""
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.lenses = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_model, bias=True)
            for _ in range(n_layers)
        ])
        for lens in self.lenses:
            torch.nn.init.eye_(lens.weight)
            torch.nn.init.zeros_(lens.bias)

    def forward(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply lens[layer_idx], then model's LN + unembed -> logits."""
        corrected = self.lenses[layer_idx](hidden)
        return model.unembed(model.ln_final(corrected))

lens = TunedLens(d_model=model.cfg.d_model, n_layers=model.cfg.n_layers)
optimizer = torch.optim.Adam(lens.parameters(), lr=1e-3)

print("initialized model and tuned lens")


# Get a dataset to train the tuned lens on. The Pile is a common choice for language modeling tasks.
import datasets
datasets.disable_caching()
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train[:1]")
print("imported dataset")

from transformers import GPT2Tokenizer
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


texts = [example["text"] for example in tqdm(ds, desc="Loading")]

tokenized_ds = []
for text in tqdm(texts, desc="Tokenizing"):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0].tolist()
    tokenized_ds.append({"tokens": tokens})



#for each layer of the model train a tuned lens using our tokenized dataset
print("starting training")
print("number of layers: ",model.cfg.n_layers)
for i in range(model.cfg.n_layers):
    
    for example in tokenized_ds:
        tokens = torch.tensor(example["tokens"]).to(device)
        with torch.no_grad():
            #extract residual stream for current data example at current layer
            _, cache = model.run_with_cache(tokens.unsqueeze(0))
            hidden = cache["resid_post", i].squeeze(0)  # (seq_len, d_model)

            #output of the model if we skip every layer except this one and pass it through to the hidden and unembedding
            target = model.unembed(model.ln_final(hidden))  # (seq_len, vocab_size)

        #use lens corresponding to layer i
        preds = lens(hidden, i)
        loss = nn.KLDivLoss(reduction="batchmean")(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("finished layer ",i)




#plotting

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

os.makedirs("plots", exist_ok=True)

sample_text = "The sky is blue and the trees are"
def plot_tuned_evolution(sample_tokens: str):
    sample_tokens = model.to_tokens(sample_text).to(device)
    token_strs = [model.to_string([t.item()]) for t in sample_tokens[0]]

    n_layers = model.cfg.n_layers
    seq_len = sample_tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(sample_tokens)
        
        top_tokens = np.empty((n_layers, seq_len), dtype=object)  # top predicted token string
        top_probs  = np.zeros((n_layers, seq_len))                # its probability

        for i in range(n_layers):
            hidden = cache["resid_post", i].squeeze(0)
            logits = lens(hidden, i)
            probs = torch.softmax(logits, dim=-1)
            top_p, top_id = probs.topk(1, dim=-1)

            for pos in range(seq_len):
                top_tokens[i, pos] = model.to_string([top_id[pos, 0].item()])
                top_probs[i, pos]  = top_p[pos, 0].item()

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
    ax.set_title(f"Tuned Lens: Top Predicted Token per Layer\nPrompt: \"{sample_text}\"", pad=20)
    plt.tight_layout()
    plt.savefig("plots/tuned_lens.svg", dpi=150, bbox_inches="tight")
    plt.show()

    # Annotate each cell with the top predicted token
    for i in range(n_layers):
        for pos in range(seq_len):
            ax.text(pos, i, repr(top_tokens[i, pos]),
                    ha="center", va="center", fontsize=7,
                    color="black" if top_probs[i, pos] < 0.7 else "white")

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([repr(t) for t in token_strs], fontsize=8, color="darkred")
    ax.set_xlabel("Input token position", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title("Tuned Lens: Top Predicted Token per Layer", fontsize=12)

    plt.tight_layout()
    plt.savefig("plots/tuned_lens.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    prompt = "The sky is blue the trees are"
    plot_tuned_evolution(prompt)