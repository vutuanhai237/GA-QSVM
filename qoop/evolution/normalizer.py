import qiskit
from .divider import by_depth as divider_by_depth
from .divider import by_num_cnot as divider_by_num_cnot
from .divider import by_num_rotation_gate as divider_by_num_rotation_gate


def by_num_cnot(num_cnot: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired number of CNOT gates

    Args:
        - qc (qiskit.QuantumCircuit)
        - selected_num_cnot (int)

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    def by_num_cnot_func(qc: qiskit.QuantumCircuit):
        qc1, _ = (divider_by_num_cnot(num_cnot))(qc)
        return qc1
    return by_num_cnot_func


def by_depth(depth: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired depth value

    Args:
        - qc (qiskit.QuantumCircuit)
        - selected_depth (int)

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    def by_depth_func(qc: qiskit.QuantumCircuit):
        if qc.depth() <= depth:
            return qc
        else:
            qc1, _ = (divider_by_depth(depth))(qc)
            return qc1
    return by_depth_func


def by_num_rotation_gate(num_rotation: int) -> qiskit.QuantumCircuit:
    """Crop circuit until achieve desired number of rotation gates (rx, ry, rz)

    Args:
        - qc (qiskit.QuantumCircuit)
        - num_rotation (int): Maximum number of rotation gates allowed in the circuit

    Returns:
        - qiskit.QuantumCircuit: Truncated circuit
    """
    def by_num_rotation_gate_func(qc: qiskit.QuantumCircuit):
        # Nếu số lượng rotation gates trong mạch nhỏ hơn hoặc bằng giới hạn, trả về mạch ban đầu
        rotation_count = len(qc.parameters)
        if rotation_count == num_rotation:
            return qc
        elif rotation_count < num_rotation:
            num_qubits = len(range(qc.num_qubits))
            thetas = qiskit.circuit.ParameterVector('eta', num_rotation - rotation_count)
            j = 0
            for i in range(rotation_count, num_rotation):
                qc.rx(thetas[j], i % num_qubits)
                j += 1
            return qc
        else:
            qc1, _ = (divider_by_num_rotation_gate(num_rotation))(qc)
        return qc1

    return by_num_rotation_gate_func

