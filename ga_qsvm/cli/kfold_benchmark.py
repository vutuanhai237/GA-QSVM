import argparse

from ga_qsvm.experiments.kfold_benchmark import run_kfold_benchmark


def build_parser():
    parser = argparse.ArgumentParser(description="Run frozen circuit k-fold benchmarks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--preprocess", choices=["legacy", "paper"], default="legacy")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rbf", "fixed-fqk", "fixed-pqk", "ga-fqk", "ga-pqk"],
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-features", type=int, default=7)
    parser.add_argument(
        "--feature-dim-mode",
        choices=["global", "circuit-parameters"],
        default="global",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_kfold_benchmark(
        manifest=args.manifest,
        seeds=args.seeds,
        folds=args.folds,
        max_folds=args.max_folds,
        preprocess=args.preprocess,
        models=args.models,
        output_dir=args.output_dir,
        n_features=args.n_features,
        feature_dim_mode=args.feature_dim_mode,
        datasets=args.datasets,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
