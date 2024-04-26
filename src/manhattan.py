import operator
import numpy as np
from functools import reduce

import typing as T

def array_to_pos_center_coordinates(X: np.ndarray) -> T.List[T.Tuple[int, int]]:
    # Helper that extracts the positive center coordinates from a numpy array
    coordinates_nonzero = np.nonzero(X > 0)

    # nonzero gives ([x, x, x], [y, y, y]), zip transposes to get (x, y), (x, y), (x, y)
    return zip(*coordinates_nonzero)


def manhattan_neighborhood(
    coor: T.Tuple[int, int],
    radius: int,
    prune_wraparound: bool=True,
    size: T.Tuple[int, int]=(11, 11)
) -> T.Set[T.Tuple[int, int]]:
    """
    Generate the set of coordinates around a center point within a given Manhattan/L1 radius. The parameter
    `prune_wraparound` determines if coordinates that fall outside the given array size are dropped. If pruning is not
    used, coordinates returned may be negative or out of range.

    NOTE that by design this neighborhood will not include the center point.

    Coordinate centers are assumed to lie within bounds.

    :param coor: Pair of integers representing the coordinates of the center point.
    :param radius: Manhattan radius of the neighborhood about the center point. Must be a positive integer.
    :param prune_wraparound: Prune points in the neighborhood that do not fit within the grid.
    :param size: Pair giving the dimensions of the overall array.
    :return: Set of all coordinates around the center point within the given Manhattan/L1 radius.
    """

    x, y = coor

    # We use a Set to automatically discard repeats. Later we can also naturally union the neighborhood around each
    # center to automatically get a list of unique cells.
    neighborhood = set()

    # Scan the L-infinity neighborhood of the center and add points that fall within the given radius.
    # We only need to scan the upper right quadrant and then add the other 3 points using symmetry. Set will handle
    # overlap.
    for d_x in range(0, radius + 1):
        for d_y in range(0, radius + 1):
            if radius >= d_x + d_y > 0:  # > 0 excludes center
                neighborhood.update(
                    [
                        (x + d_x, y + d_y),
                        (x - d_x, y + d_y),
                        (x + d_x, y - d_y),
                        (x - d_x, y - d_y),
                    ]
                )

    # Some cells in neighborhoods may be off the edges of the array. Per the problem statement we want to ignore those.
    if prune_wraparound:
        # Bounds subtract 1 because of zero indexing
        x_max, y_max = size[0] - 1, size[1] - 1
        return {
            (x, y) for (x, y) in neighborhood if (0 <= x <= x_max) & (0 <= y <= y_max)
        }

    else:
        return neighborhood

def count_positive_neighborhood_size(X: np.ndarray, radius: int, wraparound: bool=False) -> int:
    """
    Count the number of cells in a grid within a given Manhattan radius of any positive value in the array. Actual
    values in the input array are ignored, it only matters if a given entry is positive. Positive center points are
    included in the count.

    Beware that values of zero are by definition *not* positive.

    :param X: 2D Numpy array of values. They can be any type as long as they can be compared greater than 0.
    :param radius: Radius of Manhattan neighborhood around each positive center point.
    :param wraparound: Points outside of the grid are discarded by default, if true wrap them around instead.
    :return: Integer count of the number of cells in the input array within given Manhattan distance to a positive
     entry.
    """

    # Check some basic problem assumptions
    if len(X.shape) != 2:
        raise ValueError("X must be a 2-dimensional array")

    # ...
    # Not checking wacko cases like zero dimension axes etc. There are other pathological cases you can enumerate as
    # needed.

    positive_centers = list(array_to_pos_center_coordinates(X))

    # Catching case where X is all zero/negative otherwise it's tricky to handle with Python's iterator typing being
    # weird.
    if len(positive_centers) == 0:
        return 0

    all_neighbors = reduce(
        operator.or_, # or_ for Sets is union
        [
            manhattan_neighborhood(center, radius, prune_wraparound=(not wraparound),  size=X.shape)
            for center in positive_centers
        ]
    )

    # The neighborhood finder algorithm excludes centers on purpose, so now add them in.
    # Note that it's possible for a center point to fall within the neighborhood of another center so it already may be
    # `all_neighbors`, the Set object enforces uniqueness so that we get no duplicates.
    all_neighbors.update(positive_centers)

    x_max, y_max = X.shape

    if wraparound:
        # coordinates over the edges get wrapped around
        all_neighbors = set([
            (x % x_max, y % y_max) for (x, y) in all_neighbors
        ])

    return len(all_neighbors)