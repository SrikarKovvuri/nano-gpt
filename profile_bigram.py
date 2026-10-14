"""Profile the operations within bigram generation on CPU.

Put this beside data.py, model.py, and bigram.pt. Example commands:
    python profile_bigram.py --mode brute --prompt-length 5000 --tokens 100
    python profile_bigram.py --mode optimized --prompt-length 5000 --tokens 100

Use benchmark_bigram.py for real latency comparisons: profilers add overhead.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from data import encode, decode, vocab_size
from bigram_model import BigramLanguageModel


@torch.inference_mode()
def generate_profiled(model, prompt, max_new_tokens, mode):
    with record_function("01_tokenize_and_tensor"):
        #context is just the prompt converted into token ids
        ids = encode(prompt)
        context = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        with record_function("02_embedding_forward"):
            # Brute force versus optimized input: ONLY difference.
            model_input = context if mode == "brute" else context[:, -1:]
            logits = model(model_input)

        with record_function("03_select_last_logits"):
            #we only care about the last one
            logits = logits[:, -1, :]

        with record_function("04_softmax"):
            probs = F.softmax(logits, dim=-1)

        with record_function("05_multinomial_sampling"):
            #pick the next token
            next_token = torch.multinomial(probs, num_samples=1)

        with record_function("06_append_tensor_cat"):
            context = torch.cat([context, next_token], dim=1)

    with record_function("07_decode"):
        return decode(context[0, len(ids):].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["brute", "optimized"], default="brute")
    parser.add_argument("--prompt-length", type=int, default=5000)
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    if args.prompt_length < 1 or args.tokens < 1:
        parser.error("prompt-length and tokens must be >= 1")

    prompt = ("Romeo" * ((args.prompt_length + 4) // 5))[:args.prompt_length]

    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load("bigram.pt", map_location="cpu", weights_only=True))
    model.eval()

    # Warmup without profiling (ensures first-run overhead isn't included).
    torch.manual_seed(42)
    generate_profiled(model, prompt, args.tokens, args.mode)

    # The profiler tracks CPU PyTorch ops (aten::...) and our custom scopes.
    # It adds overhead, so use benchmark_bigram.py for real end-to-end timing.
    torch.manual_seed(42)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        _ = generate_profiled(model, prompt, args.tokens, args.mode)

    print(f"\nMODE={args.mode} | PROMPT={args.prompt_length} chars | GENERATED={args.tokens}")
    print("\n===== Operations sorted by total CPU time =====")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    print("\n===== Operations sorted by self CPU time =====")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    trace_path = f"bigram_trace_{args.mode}_{args.prompt_length}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace written to {trace_path}")
    print("Open the trace with https://ui.perfetto.dev (load the local file).")
    print("Compare the 02_embedding_forward and 06_append_tensor_cat scopes across modes.")


if __name__ == "__main__":
    main()
