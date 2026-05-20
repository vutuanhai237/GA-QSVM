import argparse

from ga_qsvm.experiments.pca_analysis import run_pca_analysis


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Figure 3 PCA provenance.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.95])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_pca_analysis(
        datasets=args.datasets,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
        skip_missing=args.skip_missing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
