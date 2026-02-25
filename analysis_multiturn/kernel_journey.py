#!/usr/bin/env python3
"""
Full multi-turn journey visualization for Qwen3 kernel traces.

For every kernel that achieved correctness (or all kernels with --all),
prints the complete conversation flow:

  1. System prompt (truncated)
  2. User prompt (the PyTorch code to convert)
  3. For each turn:
     - Model's generated Triton code
     - Benchmark result (correctness, speedup, error)
     - Feedback given back to the model
  4. Final chosen kernel + speedup metrics

Usage:
  python kernel_journey.py                          # correct kernels only
  python kernel_journey.py --all                    # all kernels
  python kernel_journey.py --kernel "9_Matmul"      # filter by name substring
  python kernel_journey.py --output journey.txt     # write to file

Options:
  --all              Show all kernels, not just correct ones
  --kernel SUBSTR    Only show kernels whose name contains SUBSTR
  --output FILE      Write output to FILE instead of stdout
  --no-code          Skip printing full triton code (show summary only)
  --trace FILE       Path to trace JSON (default: auto-detect)
"""

import json
import sys
import os
import argparse
import textwrap


def load_traces(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def truncate(text: str, max_lines: int = 30, label: str = "") -> str:
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    head = "\n".join(lines[:max_lines])
    return f"{head}\n    ... [{len(lines) - max_lines} more lines{' in ' + label if label else ''}] ..."


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def code_block(code: str, max_lines: int = 50, label: str = "") -> str:
    """Format code in a fenced block, optionally truncated."""
    code = code.strip()
    lines = code.splitlines()
    total = len(lines)
    if max_lines and total > max_lines:
        shown = "\n".join(lines[:max_lines])
        return (f"    ```python\n{indent(shown)}\n"
                f"    ... [{total - max_lines} more lines]\n    ```")
    return f"    ```python\n{indent(code)}\n    ```"


def format_result(result: dict) -> str:
    parts = []
    if result["correctness"]:
        parts.append("CORRECT")
    else:
        parts.append("INCORRECT")

    sp = result["speedup"]
    if sp > 0:
        parts.append(f"speedup={sp:.4f}x")
    else:
        parts.append("speedup=N/A")

    ref = result.get("reference_time_ms")
    kern = result.get("kernel_time_ms")
    if ref is not None:
        parts.append(f"ref_time={ref:.3f}ms")
    if kern is not None:
        parts.append(f"kernel_time={kern:.3f}ms")

    tiers = []
    if result["fast_0"]: tiers.append("fast_0")
    if result["fast_1"]: tiers.append("fast_1")
    if result["fast_2"]: tiers.append("fast_2")
    if tiers:
        parts.append(f"tiers=[{', '.join(tiers)}]")

    return "  |  ".join(parts)


def format_error(error: str | None, max_lines: int = 10) -> str:
    if not error:
        return ""
    lines = error.strip().splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines]"
    return error.strip()


def print_journey(d: dict, show_code: bool = True, out=sys.stdout):
    w = 80
    name = d["name"] or d["sample_key"]
    fr = d["final_result"]

    # ── Header ──
    print("=" * w, file=out)
    print(f"  KERNEL JOURNEY: {name}", file=out)
    print(f"  sample_key={d['sample_key']}  level={d.get('level', '?')}  "
          f"source={d.get('source', '?')}", file=out)
    print(f"  turns={d['num_turns']}  stop_reason={d['stop_reason']}", file=out)
    final_status = "CORRECT" if fr["correctness"] else "FAILED"
    print(f"  final_result: {final_status}  speedup={fr['speedup']:.4f}x  "
          f"fast_0={fr['fast_0']}  fast_1={fr['fast_1']}  fast_2={fr['fast_2']}", file=out)
    print("=" * w, file=out)

    # ── System prompt ──
    msgs = d.get("full_messages", [])
    system_msg = next((m for m in msgs if m["role"] == "system"), None)
    if system_msg:
        print(f"\n{'─'*w}", file=out)
        print("  [SYSTEM PROMPT]", file=out)
        print(f"{'─'*w}", file=out)
        print(indent(truncate(system_msg["content"], max_lines=20, label="system prompt")), file=out)

    # ── User prompt (original PyTorch code) ──
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    if user_msg:
        print(f"\n{'─'*w}", file=out)
        print("  [USER PROMPT] — PyTorch code to convert", file=out)
        print(f"{'─'*w}", file=out)
        print(indent(truncate(user_msg["content"], max_lines=60, label="user prompt")), file=out)
    elif d.get("pytorch_code"):
        print(f"\n{'─'*w}", file=out)
        print("  [PYTORCH CODE]", file=out)
        print(f"{'─'*w}", file=out)
        if show_code:
            print(code_block(d["pytorch_code"], max_lines=60), file=out)
        else:
            lines = d["pytorch_code"].strip().splitlines()
            print(f"    ({len(lines)} lines of PyTorch code)", file=out)

    # ── Each turn ──
    turns = sorted(d["turns"], key=lambda t: t["turn"])
    for t in turns:
        turn_num = t["turn"]
        r = t["result"]
        is_correct = r["correctness"]
        icon = "✓" if is_correct else "✗"

        print(f"\n{'━'*w}", file=out)
        print(f"  [{icon}] TURN {turn_num}", file=out)
        print(f"{'━'*w}", file=out)

        # Model reasoning (if present)
        if t.get("reasoning"):
            print(f"\n  [Model Reasoning]", file=out)
            print(indent(truncate(t["reasoning"], max_lines=30, label="reasoning")), file=out)

        # Model's full completion text (includes reasoning before <triton> tag)
        full_completion = t.get("full_completion", "")
        # Extract any text before the <triton> tag as the model's thinking
        if full_completion and "<triton>" in full_completion:
            thinking_part = full_completion.split("<triton>")[0].strip()
            if thinking_part:
                print(f"\n  [Model Thinking / Analysis]", file=out)
                print(indent(truncate(thinking_part, max_lines=40, label="model thinking")), file=out)

        # Generated Triton code
        triton_code = t.get("triton_code", "")
        if triton_code:
            print(f"\n  [Generated Triton Code]", file=out)
            if show_code:
                print(code_block(triton_code, max_lines=80), file=out)
            else:
                lines = triton_code.strip().splitlines()
                print(f"    ({len(lines)} lines of Triton code)", file=out)

        # Benchmark result
        print(f"\n  [Benchmark Result]", file=out)
        print(f"    {format_result(r)}", file=out)

        # Error details
        if r.get("error"):
            print(f"\n  [Error]", file=out)
            print(indent(format_error(r["error"], max_lines=8)), file=out)

        # Feedback given to model for next turn
        if t.get("feedback_given"):
            print(f"\n  [Feedback → Model]", file=out)
            print(indent(truncate(t["feedback_given"], max_lines=15, label="feedback")), file=out)
        elif not t.get("feedback_given") and turn_num < d["num_turns"]:
            print(f"\n  [Feedback → Model]", file=out)
            print(f"    (no explicit feedback recorded)", file=out)

    # ── Final chosen kernel ──
    print(f"\n{'━'*w}", file=out)
    print(f"  FINAL CHOSEN KERNEL", file=out)
    print(f"{'━'*w}", file=out)
    print(f"  Status     : {final_status}", file=out)
    print(f"  Speedup    : {fr['speedup']:.4f}x", file=out)
    if fr.get("reference_time_ms") is not None:
        print(f"  Ref time   : {fr['reference_time_ms']:.3f} ms", file=out)
    if fr.get("kernel_time_ms") is not None:
        print(f"  Kernel time: {fr['kernel_time_ms']:.3f} ms", file=out)
    print(f"  Tiers      : fast_0={fr['fast_0']}  fast_1={fr['fast_1']}  fast_2={fr['fast_2']}", file=out)

    if d.get("final_triton_code") and show_code:
        print(f"\n  [Final Triton Code]", file=out)
        print(code_block(d["final_triton_code"], max_lines=80), file=out)

    # ── Speedup trajectory ──
    correct_turns = [(t["turn"], t["result"]["speedup"])
                     for t in turns if t["result"]["correctness"] and t["result"]["speedup"] > 0]
    if len(correct_turns) > 1:
        print(f"\n  [Speedup Trajectory]", file=out)
        best_sp = max(s for _, s in correct_turns)
        for tn, sp in correct_turns:
            bar_len = min(int(sp * 10), 50)
            marker = " ← best" if sp == best_sp and len(correct_turns) > 1 else ""
            print(f"    Turn {tn}: {'█' * bar_len} {sp:.4f}x{marker}", file=out)

    print(f"\n{'=' * w}\n", file=out)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn kernel journey viewer")
    parser.add_argument("--trace", type=str, default=None,
                        help="Path to trace JSON file")
    parser.add_argument("--all", action="store_true",
                        help="Show all kernels, not just correct ones")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Filter by kernel name substring")
    parser.add_argument("--output", type=str, default=None,
                        help="Write output to file")
    parser.add_argument("--no-code", action="store_true",
                        help="Skip full triton code listings")
    args = parser.parse_args()

    trace_file = args.trace or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reasoning_traces_qwen3_multiturn.json"
    )

    if not os.path.exists(trace_file):
        print(f"Error: trace file not found: {trace_file}")
        sys.exit(1)

    data = load_traces(trace_file)

    # Filter
    if not args.all:
        data = [d for d in data if d["final_result"]["correctness"]]
    if args.kernel:
        substr = args.kernel.lower()
        data = [d for d in data
                if substr in (d.get("name") or "").lower()
                or substr in d["sample_key"].lower()]

    if not data:
        print("No matching kernels found.")
        sys.exit(0)

    # Sort: correct first, then by sample_key
    data.sort(key=lambda x: (not x["final_result"]["correctness"], x["sample_key"]))

    out = open(args.output, "w") if args.output else sys.stdout
    show_code = not args.no_code

    print(f"\n  Showing {len(data)} kernel journey(s)\n", file=out)

    for d in data:
        print_journey(d, show_code=show_code, out=out)

    if args.output:
        out.close()
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
