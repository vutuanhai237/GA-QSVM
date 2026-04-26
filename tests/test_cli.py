from ga_qsvm.cli.train import build_parser as build_train_parser
from ga_qsvm.cli.eval import build_parser as build_eval_parser
from ga_qsvm.cli.train import main as train_main


def test_train_cli_parser_accepts_runtime_arguments():
    parser = build_train_parser()
    args = parser.parse_args(["--depth", "4", "--num-circuit", "8", "--qubits", "3", "--data", "digits"])
    assert args.depth == [4]
    assert args.num_circuit == [8]
    assert args.qubits == [3]
    assert args.data == "digits"


def test_eval_cli_parser_accepts_runtime_arguments():
    parser = build_eval_parser()
    args = parser.parse_args(["--rx", "1", "--ry", "2", "--rz", "3", "--num-qubits", "4", "--data", "wine"])
    assert args.rx == 1
    assert args.ry == 2
    assert args.rz == 3
    assert args.num_qubits == 4
    assert args.data == "wine"


def test_train_cli_main_dispatches_to_runtime_runner(monkeypatch):
    calls = []

    def fake_run_train(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("ga_qsvm.cli.train.run_train", fake_run_train)

    train_main(["--depth", "4", "--num-circuit", "8", "--qubits", "3", "--data", "digits"])

    assert calls[0]["dataset_name"] == "digits"
    assert calls[0]["depths"] == [4]
    assert calls[0]["num_circuits"] == [8]
    assert calls[0]["qubits"] == [3]
