import qiskit

from qoop.evolution.environment import EEnvironment
from qoop.evolution.environment_synthesis import MetadataSynthesis


def _parameterized_circuit():
    circuit = qiskit.QuantumCircuit(1)
    circuit.rx(qiskit.circuit.Parameter("theta"), 0)
    return circuit


def test_evolution_stops_after_50_generations_without_improvement():
    metadata = MetadataSynthesis(
        num_qubits=1,
        num_circuit=4,
        num_generation=60,
        depth=1,
    )

    def generator(_metadata):
        return _parameterized_circuit()

    def fitness(_circuit):
        return 0.0, 0.0

    def selection(circuits, _fitnesss):
        return circuits

    def crossover(circuit1, circuit2):
        return circuit1.copy(), circuit2.copy()

    def mutate(circuit):
        return circuit

    def threshold(_fitness):
        return False

    env = EEnvironment(
        metadata=metadata,
        fitness_func=fitness,
        generator_func=generator,
        selection_func=selection,
        crossover_func=crossover,
        mutate_func=mutate,
        threshold_func=threshold,
    )

    env.evol(verbose=False, mode="serial", auto_save=False)

    assert env.metadata.current_generation == 51


def test_best_circuit_snapshot_is_not_mutated_by_next_generation():
    metadata = MetadataSynthesis(
        num_qubits=1,
        num_circuit=4,
        num_generation=1,
        depth=1,
    )

    def generator(_metadata):
        return _parameterized_circuit()

    def fitness(_circuit):
        return 1.0, 0.0

    def selection(circuits, _fitnesss):
        return circuits

    def crossover(circuit1, circuit2):
        return circuit1.copy(), circuit2.copy()

    def mutate(circuit):
        circuit.rz(qiskit.circuit.Parameter(f"mutated_{len(circuit.parameters)}"), 0)
        return circuit

    def threshold(_fitness):
        return False

    env = EEnvironment(
        metadata=metadata,
        fitness_func=fitness,
        generator_func=generator,
        selection_func=selection,
        crossover_func=crossover,
        mutate_func=mutate,
        threshold_func=threshold,
    )

    env.evol(verbose=False, mode="serial", auto_save=False)

    assert env.best_circuit.num_parameters == 1
    assert env.best_circuits[0].num_parameters == 1
