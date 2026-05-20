import argparse

from ga_qsvm.experiments.artifacts import scan_artifact_roots, write_csv


def build_parser():
    parser = argparse.ArgumentParser(description="Audit GA-QSVM circuit artifacts.")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = scan_artifact_roots(args.roots)
    write_csv(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
