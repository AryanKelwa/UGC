"""
main.py
=======
Unified command-line interface (CLI) for the Enterprise LLM Fine-Tuning Pipeline.
Manages: Data Ingestion, Anonymization, Dataset Building, SFT Training, and End-to-End runs.

Usage:
    python main.py --help
    python main.py generate [--start-batch 1] [--total-batches 34]
    python main.py clean
    python main.py prepare
    python main.py train-sft [--model llama3_8b] [--training sft]
    python main.py run-all [--model llama3_8b] [--training sft]
"""

from __future__ import annotations

import argparse
import sys

from src.utils.logger import setup_logger

log = setup_logger("pipeline")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Enterprise LLM Fine-Tuning Pipeline Command-Line Tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to run",
    )

    # 1. Generate Command
    gen = subparsers.add_parser("generate", help="Synthetically generate raw lead conversation batches")
    gen.add_argument("--start-batch", type=int, help="Override starting batch index")
    gen.add_argument("--total-batches", type=int, help="Override count of batches to generate")

    # 2. Clean Command
    subparsers.add_parser("clean", help="Anonymize personal identifiers (PII) from raw batches")

    # 3. Prepare Command
    subparsers.add_parser("prepare", help="Merge interim batches and create train/val/test splits")

    # 4. Train-SFT Command
    train = subparsers.add_parser("train-sft", help="Trigger Supervised Fine-Tuning (SFT) adapters training")
    train.add_argument("--model", type=str, default="llama3_8b", help="Model config name inside configs/model/")
    train.add_argument("--training", type=str, default="sft", help="Training config name inside configs/training/")

    # 5. Run-All Command
    run_all = subparsers.add_parser("run-all", help="Sequentially run ingestion -> clean -> prepare -> train-sft")
    run_all.add_argument("--model", type=str, default="llama3_8b", help="Model config name")
    run_all.add_argument("--training", type=str, default="sft", help="Training config name")
    run_all.add_argument("--start-batch", type=int, help="Override starting batch index")
    run_all.add_argument("--total-batches", type=int, help="Override count of batches to generate")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dynamic imports to load dependencies only on demand
    try:
        from src.data.ingestion import run_generation
        from src.data.anonymization import run_cleaning
        from src.data.dataset_builder import run_preparation
        from src.training.train_sft import run_sft
        from src.pipelines.sft_pipeline import execute_end_to_end_sft_pipeline
    except ImportError as exc:
        log.error("Failed to import core modules. Verify project paths and dependencies: %s", exc)
        sys.exit(1)

    success = False
    try:
        if args.command == "generate":
            success = run_generation(
                start_batch=args.start_batch,
                total_batches=args.total_batches
            )
        elif args.command == "clean":
            success = run_cleaning()
        elif args.command == "prepare":
            success = run_preparation()
        elif args.command == "train-sft":
            from src.utils.helpers import get_project_root
            root = get_project_root()
            run_sft(
                model_config_path=root / "configs" / "model" / f"{args.model}.yaml",
                training_config_path=root / "configs" / "training" / f"{args.training}.yaml",
                dataset_config_path=root / "configs" / "dataset" / "real_estate.yaml"
            )
            success = True
        elif args.command == "run-all":
            # For run-all, execute_end_to_end_sft_pipeline handles sequential steps
            execute_end_to_end_sft_pipeline(
                model_name=args.model,
                training_type=args.training
            )
            success = True

    except KeyboardInterrupt:
        log.warning("\nPipeline cancelled by user signal.")
        sys.exit(130)
    except Exception as exc:
        log.exception("Unhandled error raised during command execution: %s", exc)
        sys.exit(1)

    if not success:
        log.error("Command '%s' finished with errors.", args.command)
        sys.exit(1)
        
    log.info("Command '%s' executed successfully.", args.command)


if __name__ == "__main__":
    main()
