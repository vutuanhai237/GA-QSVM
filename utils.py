from qiskit import qpy


def find_permutations_sum_n(n):
    """
    Generates all permutations of three non-negative integers that sum up to a given value N.

    Args:
        n: The target sum (integer).

    Returns:
        A list of tuples, where each tuple represents a permutation (x, y, z)
        such that x + y + z = n and x, y, z are non-negative integers.
        Returns an empty list if n is negative.
    """
    if n < 0:
        return []  # No permutations of non-negative numbers will sum to a negative number

    permutations_list = []
    for x in range(n + 1):  # Iterate through possible values for the first number (x)
        for y in range(n - x + 1): # Iterate through possible values for the second number (y), ensuring x + y <= n
            z = n - x - y      # Calculate the third number (z) to make the sum equal to n
            permutations_list.append((x, y, z)) # Add the permutation (x, y, z) to the list

    return permutations_list

def load_circuits(fitness_levels):
    """
    Load quantum circuits from QPY files based on fitness levels
    
    Args:
        fitness_levels: Number of fitness levels to load
    
    Returns:
        List of loaded quantum circuits
    """
    circuits = []
    for fitness in range(1, fitness_levels + 1):
        print(fitness)
        file_name = f'4qubits_FM{fitness}_fitness_2024-12-12/best_circuit.qpy'
        with open(file_name, 'rb') as fd:
            circuit = qpy.load(fd)[0]
            print(circuit)
            circuits.append(circuit)
    return circuits