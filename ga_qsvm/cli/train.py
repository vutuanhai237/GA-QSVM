import argparse


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
    return build_parser().parse_args(argv)
