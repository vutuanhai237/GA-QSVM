import argparse

from ga_qsvm.runners.train import create_train_runner


run_train = create_train_runner()


def build_parser():
    parser = argparse.ArgumentParser(description="GA-QSVM Training Parameters")
    parser.add_argument("--depth", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--num-circuit", type=int, nargs="+", default=list(range(4, 33, 4)))
    parser.add_argument("--num-generation", type=int, nargs="+", default=[100])
    parser.add_argument("--prob-mutate", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--qubits", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--training-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--data", type=str, default="wine", choices=["digits", "wine", "cancer"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_train(
        dataset_name=args.data,
        depths=args.depth,
        num_circuits=args.num_circuit,
        num_generations=args.num_generation,
        prob_mutations=args.prob_mutate,
        qubits=args.qubits,
        training_size=args.training_size,
        test_size=args.test_size,
        num_machines=args.num_machines,
        machine_id=args.id,
        start_index=args.start_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
