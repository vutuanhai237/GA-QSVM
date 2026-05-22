from qiskit.circuit import ParameterExpression

from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot


def test_generator_random_rotation_allocation_uses_one_parameter_per_qubit():
    metadata = MetadataSynthesis(
        num_qubits=5,
        num_circuit=1,
        num_generation=1,
        depth=10,
        num_cnot=7,
    )

    circuit = by_num_rotations_and_cnot(metadata)

    assert circuit.num_parameters == 5
    assert sum(circuit.count_ops().get(name, 0) for name in ("rx", "ry", "rz")) == 5


def test_generator_includes_requested_cnot_count():
    metadata = MetadataSynthesis(
        num_qubits=4,
        num_circuit=1,
        num_generation=1,
        depth=8,
        num_cnot=6,
    )

    circuit = by_num_rotations_and_cnot(metadata)

    assert circuit.count_ops().get("cx", 0) == 6


def test_generator_does_not_emit_constant_rotation_gates():
    for _ in range(20):
        metadata = MetadataSynthesis(
            num_qubits=3,
            num_circuit=1,
            num_generation=1,
            depth=6,
            num_cnot=4,
        )
        circuit = by_num_rotations_and_cnot(metadata)

        for instruction in circuit.data:
            if instruction.operation.name not in {"rx", "ry", "rz"}:
                continue
            assert all(isinstance(param, ParameterExpression) for param in instruction.operation.params)
