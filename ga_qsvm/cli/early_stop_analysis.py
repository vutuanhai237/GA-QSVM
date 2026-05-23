import argparse

from ga_qsvm.experiments.early_stop_analysis import analyze_metadata_roots


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze GA metadata to decide whether patience-based early stopping is safe."
    )
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = analyze_metadata_roots(
        roots=args.roots,
        output_dir=args.output_dir,
        patience=args.patience,
        checkpoints=args.checkpoints,
        tolerance=args.tolerance,
        min_delta=args.min_delta,
    )
    print(f"Wrote {len(result.rows)} early-stop analysis rows to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
