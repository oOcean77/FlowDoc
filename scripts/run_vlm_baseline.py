from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.vlm_baseline import run_vlm_baseline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM inference baseline or a skipped dummy baseline.")
    parser.add_argument("--input", default="data/processed/mock_qa.csv")
    parser.add_argument("--strategy", required=True, choices=["ocr_only", "image_only", "image_ocr"])
    parser.add_argument("--backend", required=True, choices=["dummy", "qwen2_5_vl", "llava"])
    parser.add_argument("--model-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--lora-adapter")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        metrics = run_vlm_baseline(
            input_path=args.input,
            strategy=args.strategy,
            backend=args.backend,
            output_path=args.output,
            model_name=args.model_name,
            max_samples=args.max_samples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            smoke_test=args.smoke_test,
            lora_adapter=args.lora_adapter,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved VLM baseline metrics to {args.output}")
    if metrics.get("skipped"):
        print(f"skipped=true reason={metrics.get('skip_reason')}")
    else:
        print(f"field_level_accuracy={metrics.get('field_level_accuracy'):.3f}")
    if args.smoke_test:
        for item in metrics.get("smoke_logs", []):
            print(
                "smoke "
                f"sample_id={item['sample_id']} "
                f"field_name={item['field_name']} "
                f"strategy={item['strategy']} "
                f"skipped={item['skipped']} "
                f"skip_reason={item['skip_reason']} "
                f"pred_answer={item['pred_answer_preview']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
