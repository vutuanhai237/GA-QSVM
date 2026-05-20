import argparse

from ga_qsvm.experiments.circuit_export import export_circuits


def build_parser():
    parser = argparse.ArgumentParser(description="Export frozen GA-QSVM circuits for Figure 5.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["txt", "qasm"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    export_circuits(manifest=args.manifest, output_dir=args.output_dir, formats=args.formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
