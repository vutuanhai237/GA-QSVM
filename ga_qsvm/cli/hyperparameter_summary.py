import argparse

from ga_qsvm.experiments.hyperparameter_summary import summarize_hyperparameter_sources


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize existing GA hyperparameter sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summarize_hyperparameter_sources(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
