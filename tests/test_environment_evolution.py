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
