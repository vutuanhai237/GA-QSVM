import numpy as np
import qiskit
import random
from ..backend import constant
from .environment_parent import Metadata
from .environment_synthesis import MetadataSynthesis
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate

def initialize_random_parameters(num_qubits: int, max_operands: int, conditional: bool, seed):
    if max_operands < 1 or max_operands > 3:
        raise qiskit.circuit.exceptions.CircuitError("max_operands must be between 1 and 3")

    qr = qiskit.circuit.QuantumRegister(num_qubits, 'q')
    qc = qiskit.circuit.QuantumCircuit(num_qubits)

    if conditional:
        cr = qiskit.circuit.ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(seed)
    thetas = qiskit.circuit.ParameterVector('theta')
    return qr, qc, rng, thetas
    
    
def choice_from_array(arr, condition):
    item = None
    while item is None:
        item = random.choice(arr)
        if condition(item):
            return item
        else:
            item = None
    return item

def by_depth(metadata: Metadata) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    pool = constant.operations
    conditional = False
    seed=None
    max_operands = 2
    qr, qc, rng, thetas = initialize_random_parameters(num_qubits, max_operands, conditional, seed)
    thetas_length = 0
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = choice_from_array(
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], lambda value: value <= max_possible_operands)
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            num_op_pool = [
                item for item in pool if item['num_op'] == num_operands]

            operation = rng.choice(num_op_pool)
            num_params = operation['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            register_operands = [qr[i] for i in operands]
            op = operation['operation'](*angles)
            qc.append(op, register_operands)
    return qc

def weighted_choice(arr, weights):
    if len(arr) != len(weights):
        raise ValueError("The length of the arrays must be the same")
    if sum(weights) != 1:
        raise ValueError("The sum of the weights must be 1")
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return np.random.choice(arr, p=normalized_weights)

def generate_random_array(num_othergate, smallest_num_gate, n):
    # Adjust num_othergate to ensure it can be divided among n elements
    num_othergate_adjusted = num_othergate - n * smallest_num_gate
    
    if num_othergate_adjusted < 0:
        raise ValueError("num_othergate is too small to meet the condition with n elements.")
    
    # Generate an array of n elements, each initialized to smallest_num_gate
    arr = [smallest_num_gate] * n
    
    # Distribute the remaining sum (num_othergate_adjusted) randomly among the elements
    for i in range(n):
        arr[i] += random.randint(0, num_othergate_adjusted)
        num_othergate_adjusted -= arr[i] - smallest_num_gate
        
        if num_othergate_adjusted <= 0:
            break
    
    # Adjust to ensure the total sum is exactly num_othergate
    current_sum = sum(arr)
    while current_sum < num_othergate:
        arr[random.randint(0, n-1)] += 1
        current_sum += 1
    random.shuffle(arr)
    return arr

def by_num_cnot(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    pool = constant.operations_only_cnot
    # H,S,CX,RX,RY,RZ
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    num_cnot = metadata.num_cnot
    num_gate = depth * num_qubits
    num_othergate = num_gate - num_cnot
    smallest_num_gate = 1 if int(0.15 * num_othergate) == 0 else int(0.15 * num_othergate)
    # Generate random num_gates for first 4 gates
    num_gates = generate_random_array(num_othergate, smallest_num_gate, 4)
    
    num_gates.append(num_cnot)

    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    full_pool = []
    if len(pool) != len(num_gates):
        raise ValueError("The number of gates and the number of gates to be generated must be the same")
    for j, gate in enumerate(pool):
        full_pool.extend([gate]*num_gates[j])
    random.shuffle(full_pool)
    while(len(full_pool) > 0):
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)
        while(len(remaining_qubits) > 1):
            if len(full_pool) == 0:
                return qc
            gate = full_pool[-1]
            if gate['num_op'] > len(remaining_qubits):
                break
            else:
                gate = full_pool.pop()
            operands = remaining_qubits[:gate['num_op']]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]

            num_params = gate['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            op = gate['operation'](*angles)
            qc.append(op, operands)
    return qc

##############################################################################################
################################### ADDITIONAL RX RY RZ ######################################
##############################################################################################

def by_num_rotations(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    num_rx = metadata.num_rx
    num_ry = metadata.num_ry
    num_rz = metadata.num_rz
    total_rotations = num_rx + num_ry + num_rz
    num_other_gates = depth * num_qubits - total_rotations

    if total_rotations > depth * num_qubits:
        raise ValueError("Total rotation gates exceed maximum circuit gates.")

    # Pool with required gates
    rotation_gate_count = {
        "RXGate": num_rx,
        "RYGate": num_ry,
        "RZGate": num_rz,
    }

    pool = [{"operation": getattr(qiskit.circuit.library, gate_type), "num_op": 1}
            for gate_type, count in rotation_gate_count.items() for _ in range(count)]

    # Fill the rest with other gates
    other_gate_pool = [gate for gate in constant.operations_with_rotations
                       if gate['operation'] not in [RXGate,RYGate,RZGate]]
    
    for _ in range(num_other_gates):
        pool.append(random.choice(other_gate_pool))

    random.shuffle(pool)

    # Circuit creation
    qc = qiskit.QuantumCircuit(num_qubits)
    rotation_index = 1
    for gate_info in pool:
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)
        operands = remaining_qubits[:gate_info["num_op"]]
        # Add parameterized gate
        if gate_info["operation"] in [RXGate,RYGate,RZGate]:
            theta_param = qiskit.circuit.Parameter(f"theta({rotation_index})")
            rotation_index += 1
            qc.append(gate_info["operation"](theta_param), operands)
        else:
            qc.append(gate_info["operation"](), operands)
    return qc

def by_num_rotations_and_cnot(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    num_rx = metadata.num_rx
    num_ry = metadata.num_ry
    num_rz = metadata.num_rz
    num_cnot = metadata.num_cnot
    total_rotations = num_rx + num_ry + num_rz
    total_gates = depth * num_qubits
    num_other_gates = total_gates - (total_rotations + num_cnot)

    if total_rotations + num_cnot > total_gates:
        raise ValueError("The total number of specified gates exceeds the maximum allowed.")

    rotation_gate_count = {
        RXGate: num_rx,
        RYGate: num_ry,
        RZGate: num_rz,
    }
    cnot_gate_pool = [{"operation": qiskit.circuit.library.CXGate, "num_op": 2}] * num_cnot

    rotation_pool = [{"operation": gate_type, "num_op": 1}
                     for gate_type, count in rotation_gate_count.items() for _ in range(count)]

    smallest_num_gate = 1 if int(0.15 * num_other_gates) == 0 else int(0.15 * num_other_gates)
    num_other_gates_pool = generate_random_array(num_other_gates, smallest_num_gate, 4)
    pool = constant.operations_with_rotations
    other_gate_pool = [gate for gate in pool if gate['operation'] not in
                       [RXGate, RYGate, RZGate, CXGate]]

    full_pool = rotation_pool + cnot_gate_pool
    for count, gate_info in zip(num_other_gates_pool, other_gate_pool):
        full_pool.extend([gate_info] * count)

    random.shuffle(full_pool)

    qc = qiskit.QuantumCircuit(num_qubits)
    rotation_index = 0  # Index for theta parameters
    while full_pool:
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)
        gate_info = full_pool.pop()
        operands = remaining_qubits[:gate_info["num_op"]]
        if gate_info["operation"] in [RXGate, RYGate, RZGate]:
            theta_param = qiskit.circuit.Parameter(f"theta({rotation_index})")
            rotation_index += 1
            qc.append(gate_info["operation"](theta_param), operands)
        else:
            qc.append(gate_info["operation"](), operands)

    return qc

