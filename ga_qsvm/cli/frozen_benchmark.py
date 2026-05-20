import argparse

from ga_qsvm.experiments.frozen_benchmark import run_frozen_benchmark


def build_parser():
    parser = argparse.ArgumentParser(description="Run frozen circuit holdout benchmarks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--preprocess", choices=["legacy", "paper"], default="legacy")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rbf", "fixed-fqk", "fixed-pqk", "ga-fqk", "ga-pqk"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-features", type=int, default=7)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_frozen_benchmark(
        manifest=args.manifest,
        seeds=args.seeds,
        test_size=args.test_size,
        preprocess=args.preprocess,
        models=args.models,
        output_dir=args.output_dir,
        n_features=args.n_features,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
