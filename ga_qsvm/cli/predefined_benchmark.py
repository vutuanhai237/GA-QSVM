import argparse

from ga_qsvm.experiments.predefined_benchmark import run_predefined_benchmark


PREDEFINED_MODELS = [
    "efficient-su2-fqk",
    "efficient-su2-pqk",
    "two-local-fqk",
    "two-local-pqk",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Run predefined circuit holdout benchmarks.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--qubits", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--preprocess", choices=["legacy", "paper"], default="paper")
    parser.add_argument("--models", nargs="+", choices=PREDEFINED_MODELS, default=PREDEFINED_MODELS)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_predefined_benchmark(
        datasets=args.datasets,
        qubits=args.qubits,
        seeds=args.seeds,
        test_size=args.test_size,
        preprocess=args.preprocess,
        models=args.models,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
