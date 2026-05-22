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
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-features", type=int, default=7)
    parser.add_argument(
        "--feature-dim-mode",
        choices=["global", "circuit-parameters"],
        default="global",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type", default="holdout-benchmark")
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
        feature_dim_mode=args.feature_dim_mode,
        datasets=args.datasets,
        wandb_config=(
            {
                "project": args.wandb_project,
                "name": args.wandb_name,
                "group": args.wandb_group,
                "job_type": args.wandb_job_type,
                "config": {
                    "manifest": args.manifest,
                    "seeds": args.seeds,
                    "test_size": args.test_size,
                    "preprocess": args.preprocess,
                    "models": args.models,
                    "datasets": args.datasets,
                    "output_dir": args.output_dir,
                    "n_features": args.n_features,
                    "feature_dim_mode": args.feature_dim_mode,
                },
            }
            if args.wandb_project
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
