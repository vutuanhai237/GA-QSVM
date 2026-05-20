import argparse

from ga_qsvm.experiments.summarize_reviewer_results import summarize_reviewer_results


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize reviewer benchmark outputs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summarize_reviewer_results(inputs=args.inputs, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
