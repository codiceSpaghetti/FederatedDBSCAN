import numpy as np
from scipy.spatial import distance
import math
import arffutils as arff


def get_all_neighbors(cell, get_corners=False):
    diag_coord = [(x - 1, x, x + 1) for x in cell]

    cartesian_product = [[]]
    for pool in diag_coord:
        cartesian_product = [(x + [y]) for x in cartesian_product for y in pool]

    neighbors = []
    for prod in cartesian_product:
        if get_corners:
            neighbors.append(tuple(prod))
        else:
            differential_coord = 0
            for i in range(len(prod)):
                if prod[i] != cell[i]:
                    differential_coord += 1
            if differential_coord == 1:
                neighbors.append(tuple(prod))

    if get_corners:
        neighbors.remove(cell) # remove the cell itself

    return neighbors


def get_points(partition_index, L, floor=False):
    data, meta = arff.loadpartition(partition_index)
    dimension = len(data[0]) - 1
    points = []
    for row in data:
        if floor:
            points.append(tuple(math.floor(row[i] / L) for i in range(dimension)))
        else:
            points.append(tuple(row[i] for i in range(dimension)))
    return points


def compute_local_update(my_index, L):
    """ Computes the local updates with a specific grid granularity and returns a mapping of the computed cells
        associated with the number of points in each cell

    :param my_index: int. Index of the partition file to read from
    :param L: float. Grid's granularity
    :return: dict. Map containing couples (cell coords) -> # of points in cell
                Example of return: {(9, 27): 2.0, (9, 29): 3.0, (9, 30): 1.0}
    """
    cells = np.array(get_points(my_index, L, floor=True))
    #print(len(cells))
    dimensions = len(cells[0])

    max_cell_coords = []
    min_cell_coords = []
    for i in range(dimensions):
        max_cell_coords.append(np.amax(cells[:, i]))
        min_cell_coords.append(np.amin(cells[:, i]))

    shifts = np.zeros(dimensions)
    for i in range(dimensions):
        if min_cell_coords[i] < 0:
            shifts[i] = -1 * min_cell_coords[i]

    shifted_dimensions = ()
    for i in range(dimensions):
        shifted_dimensions += (int(max_cell_coords[i] + 1 + shifts[i]), )

    count_matrix = np.zeros(shifted_dimensions)
    for cell in cells:
        shifted_cell_coords = ()
        for i in range(dimensions):
            shifted_cell_coords += (int(cell[i] + shifts[i]),)
        count_matrix[shifted_cell_coords] += 1

    non_zero = np.where(count_matrix > 0)
    non_zero_indexes = []
    for i in range(len(non_zero)):
        for j in range(len(non_zero[i])):
            if i == 0:
                non_zero_indexes.append((int(non_zero[i][j]), ))
            else:
                non_zero_indexes[j] += (int(non_zero[i][j]), )

    dict_to_return = {}
    for index in non_zero_indexes:
        if count_matrix[index] > 0:
            shifted_index = ()
            for i in range(len(index)):
                shifted_index += (int(index[i] - shifts[i]), )
            dict_to_return[shifted_index] = count_matrix[index]

    return dict_to_return


def assign_points_to_cluster(my_index, array_cells, labels, L):
    """ Assigns to each point of the dataset the class label associated with the cell which contains such point
        or sets it as an outlier if no such cell is been clustered.

    :param my_index: int. index indicating the dataset file
    :param array_cells: numpy.ndarray. Array of cells which have been clustered
    :param labels: numpy.ndarray. Array of class labels, each associated with the corresponding cell (the cell in cells
                    which has the same index)
    :param L: int. Granularity of the grid
    :return: points_to_return, labels_to_return.
                points_to_return: numpy.ndarray. Array of points contained in the dataset
                labels_to_return: numpy.ndarray. Array of labels, each associated to the point having the same index
    """
    points = get_points(my_index, L, floor=False)

    dense_cells = []
    for row in array_cells:
        dense_cells.append(tuple(row))

    points_to_return = []
    labels_to_return = []

    while len(points) > 0:
        actual_point = points.pop(0)
        actual_cell = tuple(math.floor(actual_point[i] / L) for i in range(len(actual_point)))
        outlier = True

        if actual_cell in dense_cells:
            points_to_return.append(actual_point)
            labels_to_return.append(labels[dense_cells.index(actual_cell)])
        else:
            min_dist = float('inf')
            cluster_to_assign = -1
            check_list = get_all_neighbors(actual_cell)
            for check_cell in check_list:
                if check_cell in dense_cells:
                    cell_mid_point = tuple(cell_coord * L + L/2 for cell_coord in check_cell)
                    actual_dist = distance.euclidean(actual_point, cell_mid_point)
                    if actual_dist < min_dist:
                        min_dist = actual_dist
                        cluster_to_assign = labels[dense_cells.index(check_cell)]

                    outlier = False

            if outlier:
                points_to_return.append(actual_point)
                labels_to_return.append(-1)
            else:
                points_to_return.append(actual_point)
                labels_to_return.append(cluster_to_assign)

    return np.array(points_to_return), np.array(labels_to_return)