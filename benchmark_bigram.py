"""Benchmark brute-force vs last-token-only bigram generation on CPU.

Put this beside data.py, model.py, and bigram.pt, then run:
    python benchmark_bigram.py

Optional arguments:
    python benchmark_bigram.py --tokens 500 --lengths 5 500 5000 10000 --trials 7
"""

import argparse
import csv
import statistics
import time

import torch
import torch.nn.functional as F

from data import encode, decode, vocab_size
from bigram_model import BigramLanguageModel


# Keep this function separate from model.py so we can test BOTH versions
# without manually editing the model between benchmark runs.
@torch.inference_mode()
def generate_for_benchmark(model, prompt, max_new_tokens, mode):
    #context is just the prompt converted into token ids
    ids = encode(prompt)  # list of ids
    context = torch.tensor([ids], dtype=torch.long)  # (1, len(prompt))

    for _ in range(max_new_tokens):
        if mode == "brute":
            model_input = context  # (1, growing sequence length)
        elif mode == "optimized":
            model_input = context[:, -1:]  # (1, 1), just the last token
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logits = model(model_input)  # (1, model_input length, 65)

        #we only care about the last one
        logits = logits[:, -1, :]  # (1, 65)

        #pick the next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        context = torch.cat([context, next_token], dim=1)

    #convert context into a list and then decode it
    return decode(context[0, len(ids):].tolist())


def run_timed_trial(model, prompt, max_new_tokens, mode, seed=42):
    # Same random seed for both methods -> same generated token samples.
    torch.manual_seed(seed)
    start = time.perf_counter()
    _ = generate_for_benchmark(model, prompt, max_new_tokens, mode)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 500, 5000, 10000])
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    if args.tokens < 1 or args.trials < 1 or args.warmups < 0 or any(n < 1 for n in args.lengths):
        parser.error("tokens, trials and prompt lengths must be >= 1; warmups must be >= 0")

    # Load the SAME trained weights for both variants.
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load("bigram.pt", map_location="cpu", weights_only=True))
    model.eval()

    print(f"Device: CPU | Generated tokens per run: {args.tokens}")
    print(f"Trials: {args.trials}, warmups: {args.warmups} | Seed: 42")
    print("Timing includes encode, PyTorch forward/sampling, torch.cat, and decode.")

    csv_rows = []
    for prompt_len in args.lengths:
        prompt = ("Romeo" * ((prompt_len + 4) // 5))[:prompt_len]

        # Correctness check: our two implementations must produce identical output
        # when the random sampling seed is identical.
        torch.manual_seed(42)
        brute_output = generate_for_benchmark(model, prompt, args.tokens, "brute")
        torch.manual_seed(42)
        optimized_output = generate_for_benchmark(model, prompt, args.tokens, "optimized")
        assert brute_output == optimized_output, "Outputs differ; benchmark isn't an apples-to-apples comparison!"

        # Warm up both methods without timing.
        for _ in range(args.warmups):
            run_timed_trial(model, prompt, args.tokens, "brute")
            run_timed_trial(model, prompt, args.tokens, "optimized")

        timings = {"brute": [], "optimized": []}

        # Alternate the order to avoid favoring whichever runs first.
        for trial in range(args.trials):
            order = ("brute", "optimized") if trial % 2 == 0 else ("optimized", "brute")
            for mode in order:
                timings[mode].append(run_timed_trial(model, prompt, args.tokens, mode))

        brute_s = statistics.median(timings["brute"])
        opt_s = statistics.median(timings["optimized"])
        speedup = brute_s / opt_s
        total_brute_lookup_positions = args.tokens * prompt_len + args.tokens * (args.tokens - 1) // 2

        print(f"\nPrompt: {prompt_len:,} chars | model input positions processed:")
        print(f"  brute={total_brute_lookup_positions:,}  optimized={args.tokens:,}")
        print(f"  brute median:      {brute_s * 1000:9.3f} ms  ({args.tokens / brute_s:,.0f} tokens/sec)")
        print(f"  optimized median:  {opt_s * 1000:9.3f} ms  ({args.tokens / opt_s:,.0f} tokens/sec)")
        print(f"  end-to-end speedup: {speedup:.3f}x")
        print(f"  brute range: {min(timings['brute'])*1000:.1f} - {max(timings['brute'])*1000:.1f} ms")
        print(f"  opt range:   {min(timings['optimized'])*1000:.1f} - {max(timings['optimized'])*1000:.1f} ms")

        csv_rows.append({
            "prompt_chars": prompt_len,
            "generated_tokens": args.tokens,
            "brute_median_ms": round(brute_s * 1000, 4),
            "optimized_median_ms": round(opt_s * 1000, 4),
            "brute_tokens_per_sec": round(args.tokens / brute_s, 2),
            "optimized_tokens_per_sec": round(args.tokens / opt_s, 2),
            "speedup_x": round(speedup, 4),
        })

    with open("bigram_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nSaved results to bigram_benchmark_results.csv")


if __name__ == "__main__":
    main()
