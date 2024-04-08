import pytest
import itertools
import numpy as np
from src import count_positive_neighborhood_size

import random

def centers_to_array(centers, width, height):
    # Problem only cares about strictly positive center values, so 0 is fine
    # the actual value if not positive is irrelevant
    X = np.zeros((height, width), dtype=int)
    for center in centers:
        X[center] = 1

    return X

@pytest.mark.parametrize(
    "centers,radius,count",
    [  # centers, radius, count
        ([(5, 5), ], 3, 25),
        ([(5, 1), ], 3, 21),
        ([(7, 3), (3, 7), ], 2, 26),
        ([(7, 3), (6, 5), ], 2, 22),
    ],
    ids=["example_1", "example_2", "example_3", "example_4"]
)
class TestBasicExamples:
    WIDTH = 11
    HEIGHT = 11

    def test_example(self, centers, radius, count):
        X = centers_to_array(centers, self.WIDTH, self.HEIGHT)
        assert count_positive_neighborhood_size(X, radius) == count


## Extended warning tests

class TestWarning1:
    WIDTH = 11
    HEIGHT = 11

    @pytest.mark.parametrize(
        "centers,radius,count",
        [  # centers, radius, count
            ([(5, 0), (1, 1)], 3, 29),
            ([(0, 0), (1, 1)], 3, 17),
        ],
        ids=["1a", "1b",],
    )
    def test_example(self, centers, radius, count):
        X = centers_to_array(centers, self.WIDTH, self.HEIGHT)
        assert count_positive_neighborhood_size(X, radius) == count


class TestWarning2:
    WIDTH = 11
    HEIGHT = 11

    @pytest.mark.parametrize(
        "centers,radius,count",
        [  # centers, radius, count
            ([(5, 5), (5, 6), (4, 5)], 3, 36),
            ([(10, 2), (10, 3), (10, 4)], 3, 23)
        ],
        ids=["2a", "2b"],
    )
    def test_example(self, centers, radius, count):
        X = centers_to_array(centers, self.WIDTH, self.HEIGHT)
        assert count_positive_neighborhood_size(X, radius) == count


class TestWarning3:
    WIDTH = 11
    HEIGHT = 11

    @pytest.mark.parametrize(
        "centers,radius,count",
        [  # centers, radius, count
            ([(10, 0), (0, 10)], 3, 20),  # also did you remember to prune wraparounds on the right?
            ([(0, 0), (0, 10), (10, 0), (10, 10)], 3, 40),
        ],
        ids=["3a", "3b"]
    )
    def test_example(self, centers, radius, count):
        X = centers_to_array(centers, self.WIDTH, self.HEIGHT)
        assert count_positive_neighborhood_size(X, radius) == count

class TestWarning4:
    @pytest.mark.parametrize(
        "height,width,centers,radius,count",
        [
            (1, 21, [(0, 5)], 3, 7),
            (1, 1, [(0, 0)], 2, 1),
            (10, 1, [(3, 0)], 2, 5),
            (2, 2, [(1, 1)], 2, 4)
        ],
        ids=["4a", "4b", "4c", "4d"],
    )
    def test_example(self, width, height, centers, radius, count):
        X = centers_to_array(centers, width, height)
        assert count_positive_neighborhood_size(X, radius) == count

class TestWarning5:
    # Scaling should depend more strongly on radius than number
    DIM = 100
    NUM = 1000

    @pytest.mark.timeout(1)
    def test_big_fast_enough(self):
        centers = zip(
            np.random.choice(self.DIM, size=self.NUM, replace=True).astype(int),
            np.random.choice(self.DIM, size=self.NUM, replace=True).astype(int),
        )
        centers = list(centers)

        X = centers_to_array(centers, self.DIM, self.DIM)
        count_positive_neighborhood_size(X, 2)

class TestWarning6:
    WIDTH = 11
    HEIGHT = 11

    @pytest.mark.parametrize(
        "centers,radius,count",
        [  # centers, radius, count
            ([], 3, 0),
        ],
        ids=["6a"]
    )
    def test_example(self, centers, radius, count):
        X = centers_to_array(centers, self.WIDTH, self.HEIGHT)
        assert count_positive_neighborhood_size(X, radius) == count

## Extra tests not mentioned in problem

def test_radius_zero():
    X = centers_to_array([(2, 1), (7, 4)], 11, 11)
    assert count_positive_neighborhood_size(X, 0) == 2

def test_square_transpose_invariant():
    # X and X.T should be the same for square arrays
    # Should also be for rectangular arrays but we're already off topic so going to omit that test.
    X1 = centers_to_array([(0, 7), (10, 8)], 11, 11)
    X2 = centers_to_array([(7, 0), (8, 10)], 11, 11)

    c1 = count_positive_neighborhood_size(X1, 2)
    c2 = count_positive_neighborhood_size(X2, 2)

    assert c1 == c2

def test_rectangular_transpose_invariant():
    # X and X.T should be the same as long as all regions are inside the bounds
    # Should also be for rectangular arrays but we're already off topic so going to omit that test.
    X1 = centers_to_array([(4, 7), (6, 8)], 17, 11)
    X2 = centers_to_array([(7, 4), (8, 6)], 11, 17)

    c1 = count_positive_neighborhood_size(X1, 2)
    c2 = count_positive_neighborhood_size(X2, 2)

    assert c1 == c2
