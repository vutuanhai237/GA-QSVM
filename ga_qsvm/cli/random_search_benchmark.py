import argparse

from ga_qsvm.experiments.random_search_benchmark import run_random_search_benchmark


RANDOM_MODELS = ["random-fqk", "random-pqk"]


def build_parser():
    parser = argparse.ArgumentParser(description="Run random circuit search holdout benchmarks.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--qubits", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--preprocess", choices=["legacy", "paper"], default="paper")
    parser.add_argument("--models", nargs="+", choices=RANDOM_MODELS, default=["random-pqk"])
    parser.add_argument("--random-budget", type=int, default=20)
    parser.add_argument("--depth-multiplier", type=int, default=5)
    parser.add_argument("--num-cnot-multiplier", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_random_search_benchmark(
        datasets=args.datasets,
        qubits=args.qubits,
        seeds=args.seeds,
        test_size=args.test_size,
        preprocess=args.preprocess,
        models=args.models,
        output_dir=args.output_dir,
        random_budget=args.random_budget,
        depth_multiplier=args.depth_multiplier,
        num_cnot_multiplier=args.num_cnot_multiplier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
