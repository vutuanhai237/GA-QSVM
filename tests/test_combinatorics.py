from ga_qsvm.utils.combinatorics import find_permutations_sum_n


def test_find_permutations_sum_n_for_three():
    assert find_permutations_sum_n(3) == [
        (0, 0, 3),
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 0),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
        (3, 0, 0),
    ]


def test_find_permutations_sum_n_for_negative_input():
    assert find_permutations_sum_n(-1) == []
