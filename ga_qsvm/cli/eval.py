import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="GA-QSVM Evaluation")
    parser.add_argument("--rx", type=int, default=4)
    parser.add_argument("--ry", type=int, default=1)
    parser.add_argument("--rz", type=int, default=2)
    parser.add_argument("--num-qubits", dest="num_qubits", type=int, default=7)
    parser.add_argument("--prob-mutate", dest="prob_mutate", type=float, default=0.027825594022071243)
    parser.add_argument("--data", type=str, default="digits", choices=["digits", "wine", "cancer"])
    return parser


def main(argv=None):
    return build_parser().parse_args(argv)
