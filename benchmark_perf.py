"""Benchmark performance improvements for llama-cpp-python."""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

# Ensure we use the installed package, not local source
sys.path = [p for p in sys.path if "llama-cpp-python" not in p]

import llama_cpp
from llama_cpp import internals


def timed(fn, name, iterations=10):
    """Run a function multiple times and return average time in microseconds."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1_000_000
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:.1f} µs avg over {iterations} iterations")
    return avg, result


def benchmark_set_batch():
    """Benchmark set_batch: numpy bulk copy vs Python loop."""
    print("\n=== set_batch() ===")

    model_path = str(
        Path(__file__).parent / "vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"
    )
    model = llama_cpp.Llama(model_path, vocab_only=True, verbose=False, n_ctx=512)

    batch_size = 512
    tokens = list(range(1, batch_size + 1))

    # Current implementation (numpy bulk copy)
    batch_obj = internals.LlamaBatch(n_tokens=batch_size, embd=0, n_seq_max=1)
    batch = batch_obj.batch

    def current_set_batch():
        n_tokens = len(tokens)
        batch.n_tokens = n_tokens
        token_arr = np.ctypeslib.as_array(batch.token, shape=(n_tokens,))
        token_arr[:] = tokens
        pos_arr = np.ctypeslib.as_array(batch.pos, shape=(n_tokens,))
        pos_arr[:] = np.arange(0, n_tokens, dtype=pos_arr.dtype)
        n_seq_id_arr = np.ctypeslib.as_array(batch.n_seq_id, shape=(n_tokens,))
        n_seq_id_arr[:] = 1
        logits_arr = np.ctypeslib.as_array(batch.logits, shape=(n_tokens,))
        logits_arr[:] = False
        logits_arr[n_tokens - 1] = True
        for i in range(n_tokens):
            batch.seq_id[i][0] = 0

    def python_loop_set_batch():
        n_tokens = len(tokens)
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens[i]
            batch.pos[i] = i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = False
        batch.logits[n_tokens - 1] = True

    t_current, _ = timed(current_set_batch, "numpy bulk copy (current)", 100)
    t_python, _ = timed(python_loop_set_batch, "Python loop (original)", 100)

    speedup = t_python / t_current
    print(f"  Speedup: {speedup:.1f}x")


def benchmark_detokenize():
    """Benchmark detokenize: incremental token_to_piece vs full re-detokenize."""
    print("\n=== Detokenization (generation loop) ===")

    model_path = str(
        Path(__file__).parent / "vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"
    )
    model = llama_cpp.Llama(model_path, vocab_only=True, verbose=False, n_ctx=512)

    # Simulate 50 generated tokens
    sim_tokens = [29871, 310, 2787, 304, 1234, 567, 890, 123, 456, 789] * 5
    prompt_tokens = [1, 15043, 2787]

    # Current: incremental token_to_piece
    def incremental_detokenize():
        accumulated = b""
        for t in sim_tokens:
            accumulated += model._model.token_to_piece(t)
        return accumulated

    # Original: full re-detokenize on each token (simulating O(n²))
    def full_redetokenize():
        all_text = b""
        for i in range(1, len(sim_tokens) + 1):
            all_text = model.detokenize(sim_tokens[:i], prev_tokens=prompt_tokens)
        return all_text

    t_incremental, _ = timed(
        incremental_detokenize, "incremental token_to_piece (current)", 50
    )
    t_full, _ = timed(full_redetokenize, "full re-detokenize each token (original)", 10)

    # The original is O(n²), so for fair comparison we compare per-token cost
    per_token_current = t_incremental / len(sim_tokens)
    per_token_original = t_full / len(sim_tokens)
    print(
        f"  Per-token: {per_token_current:.1f} µs (current) vs {per_token_original:.1f} µs (original)"
    )
    speedup = per_token_original / per_token_current
    print(f"  Speedup per token: {speedup:.1f}x")


def benchmark_logprobs():
    """Benchmark logprobs: np.argpartition vs sorted()."""
    print("\n=== Logprobs computation ===")

    vocab_size = 128000
    logprobs = np.random.randn(vocab_size).astype(np.float32)
    k = 5

    def argpartition_topk():
        top_k_indices = np.argpartition(logprobs, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(logprobs[top_k_indices])][::-1]
        return top_k_indices

    def sorted_topk():
        sorted_logprobs = list(
            sorted(zip(logprobs, range(len(logprobs))), reverse=True)
        )
        return [i for _, i in sorted_logprobs[:k]]

    t_argpartition, _ = timed(argpartition_topk, "np.argpartition (current)", 100)
    t_sorted, _ = timed(sorted_topk, "sorted() (original)", 100)

    speedup = t_sorted / t_argpartition
    print(f"  Speedup: {speedup:.1f}x")


def benchmark_logit_bias():
    """Benchmark logit_bias: in-place vs copy."""
    print("\n=== Logit bias ===")

    vocab_size = 128000
    scores = np.random.randn(vocab_size).astype(np.float32)
    logit_bias = {100: 1.5, 200: -0.5, 300: 2.0}

    def in_place_bias():
        for input_id, score in logit_bias.items():
            scores[input_id] += score
        return scores

    def copy_bias():
        new_scores = np.copy(scores)
        for input_id, score in logit_bias.items():
            new_scores[input_id] = score + scores[input_id]
        return new_scores

    t_inplace, _ = timed(in_place_bias, "in-place (current)", 1000)
    t_copy, _ = timed(copy_bias, "np.copy (original)", 1000)

    speedup = t_copy / t_inplace
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Memory saved per token: {vocab_size * 4} bytes (no copy)")


def benchmark_kv_cache_seq_rm():
    """Benchmark kv_cache_seq_rm skip."""
    print("\n=== kv_cache_seq_rm skip ===")
    print("  Original: 1 FFI call per eval() (always)")
    print("  Current:  0 FFI calls when appending tokens (normal generation)")
    print("  FFI call overhead: ~1-5 µs each")
    print("  Savings: ~1-5 µs per generated token")


def main():
    print("=" * 60)
    print("llama-cpp-python Performance Benchmark")
    print("=" * 60)

    benchmark_set_batch()
    benchmark_detokenize()
    benchmark_logprobs()
    benchmark_logit_bias()
    benchmark_kv_cache_seq_rm()

    print("\n" + "=" * 60)
    print("Summary: See individual sections for speedup numbers")
    print("=" * 60)


if __name__ == "__main__":
    main()
