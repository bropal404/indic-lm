import sys
import torch

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CHECKPOINT = "last.pt"
TOKENIZER_FILE = "hindi_bpe16k.pkl"

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.8
TOP_K = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Make sure local model.py and tokenizers.py can be imported
sys.path.insert(0, ".")


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------
from tokenizers import customBPE
from model import GPTLanguageModel


def encode(tokenizer, text):
    ids = []

    if "<bos>" in tokenizer.vocab:
        ids.append(tokenizer.vocab["<bos>"])

    for word in tokenizer.pre_tokenizer.tokenize(text):

        # Explicit word-boundary token
        if "▁" in tokenizer.vocab:
            ids.append(tokenizer.vocab["▁"])

        pieces = tokenizer._token_cache.get(word)

        if pieces is None:
            pieces = tokenizer.tokenize(word)

        for piece in pieces:
            ids.append(
                tokenizer.vocab.get(
                    piece,
                    tokenizer.vocab.get("UNK", 0)
                )
            )

    return ids


def decode(ids, id_to_token):
    words = []
    current = ""

    for i in ids:
        token = id_to_token.get(int(i), "")

        if token == "▁":
            if current:
                words.append(current)
            current = ""

        elif token in ("<pad>", "<bos>", "<eos>"):
            continue

        else:
            current += token

    if current:
        words.append(current)

    return " ".join(words)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
tokenizer = customBPE()
tokenizer.load(TOKENIZER_FILE)

id_to_token = {
    i: token
    for token, i in tokenizer.vocab.items()
}

print(f"Tokenizer loaded: {TOKENIZER_FILE}")
print(f"Vocab size: {len(tokenizer.vocab):,}")

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

vocab_size = ckpt["vocab_size"]
d_model = ckpt["d_model"]
n_heads = ckpt["n_heads"]
n_layers = ckpt["n_layers"]
d_ff = ckpt["d_ff"]
context_len = ckpt.get("context_len", 256)
pad_token_id = ckpt.get(
    "pad_token_id",
    tokenizer.vocab.get("<pad>", 0)
)

model = GPTLanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    context_len=context_len,
    dropout=0.0,
    pad_token_id=pad_token_id,
)

model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

print(f"Model loaded: {CHECKPOINT}")
print(f"Device: {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()


# --------------------------------------------------
# GENERATION
# --------------------------------------------------
@torch.no_grad()
def generate(prompt):
    ids = torch.tensor(
        [encode(tokenizer, prompt)],
        dtype=torch.long,
        device=DEVICE,
    )

    # Greedy
    greedy = model.greedy_generate(
        ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Sampling
    sampled = model.generate(
        ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    return (
        decode(greedy[0].tolist(), id_to_token),
        decode(sampled[0].tolist(), id_to_token),
    )


# --------------------------------------------------
# REPL
# --------------------------------------------------
print("Hindi GPT REPL")
print("Type 'exit' or 'quit' to stop.")
print()

while True:
    prompt = input("> ").strip()

    if prompt.lower() in ("exit", "quit"):
        break

    if not prompt:
        continue

    greedy, sampled = generate(prompt)

    print(f"\nGreedy : {greedy}")
    print(f"Sampled: {sampled}\n")

