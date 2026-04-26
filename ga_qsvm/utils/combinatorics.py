def find_permutations_sum_n(n: int) -> list[tuple[int, int, int]]:
    if n < 0:
        return []

    permutations_list: list[tuple[int, int, int]] = []
    for x in range(n + 1):
        for y in range(n - x + 1):
            z = n - x - y
            permutations_list.append((x, y, z))
    return permutations_list
