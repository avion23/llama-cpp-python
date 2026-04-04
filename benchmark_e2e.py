"""End-to-end generation benchmark for llama-cpp-python."""

import sys
import time
import os
from pathlib import Path

# Ensure we use the installed package
sys.path = [p for p in sys.path if "llama-cpp-python" not in p]

import llama_cpp


def benchmark_generation(model_path, prompt, max_tokens=50, n_runs=3):
    """Benchmark token generation speed."""
    print(f"\nModel: {os.path.basename(model_path)}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Max tokens: {max_tokens}")
    print()

    times = []
    all_tokens = []

    for i in range(n_runs):
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=2048,
            verbose=False,
            n_threads=os.cpu_count(),
            n_threads_batch=os.cpu_count(),
        )

        start = time.perf_counter()
        output = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
        )
        elapsed = time.perf_counter() - start

        gen_tokens = output["usage"]["completion_tokens"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        text = output["choices"][0]["text"]

        times.append(elapsed)
        all_tokens.append(gen_tokens)

        print(
            f"  Run {i + 1}: {gen_tokens} tokens in {elapsed:.2f}s "
            f"({gen_tokens / elapsed:.1f} tok/s) "
            f"[prompt: {prompt_tokens} tokens]"
        )

        del model

    avg_time = sum(times) / len(times)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    print(
        f"\n  Average: {avg_tokens:.0f} tokens in {avg_time:.2f}s "
        f"({avg_tokens / avg_time:.1f} tok/s)"
    )
    print(f"  Generated text: {text[:100]}...")
    return avg_tokens / avg_time


def main():
    print("=" * 60)
    print("llama-cpp-python End-to-End Generation Benchmark")
    print("=" * 60)

    # Use the test model from the test suite
    from huggingface_hub import hf_hub_download

    repo_id = "lmstudio-community/Qwen3.5-0.8B-GGUF"
    filename = "Qwen3.5-0.8B-Q8_0.gguf"

    print(f"\nDownloading model {repo_id}/{filename}...")
    model_path = hf_hub_download(repo_id, filename)
    print(f"Model path: {model_path}")

    prompts = [
        "The quick brown fox jumps over the lazy dog. The quick brown fox",
        "Write a short story about a programmer who discovered",
        "Explain quantum computing in simple terms:",
    ]

    results = []
    for prompt in prompts:
        tok_s = benchmark_generation(model_path, prompt, max_tokens=50, n_runs=3)
        results.append((prompt[:40], tok_s))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for prompt_desc, tok_s in results:
        print(f"  {prompt_desc}...: {tok_s:.1f} tok/s")

    avg_tok_s = sum(r[1] for r in results) / len(results)
    print(f"\n  Overall average: {avg_tok_s:.1f} tok/s")


if __name__ == "__main__":
    main()
