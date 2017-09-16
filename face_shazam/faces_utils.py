# -*- coding: utf-8 -*-
import math

import numpy as np


def process_faces(all_subjects, training_percentage):
    """ Processes the given data (i.e all_subjects).

    Params:
        all_subjects (dict): The dictionary holding each subject's list of images (represented as numpy's ndarray).
        training_percentage (float): The percentage of training elements.
    Returns:

    """
    # TODO complete returns
    separated_data_set = _separate_data_set(all_subjects, training_percentage)
    training_set = separated_data_set["train"]
    difference_matrix = _get_difference_matrix(training_set)  # TODO: usar para calcular autocaras
    l_matrix = _l_matrix(difference_matrix)

    # TODO: Calcular eigenvectors
    # TODO: Calcular eigenfaces (pag. 75)

    return


def _get_difference_matrix(training_set):  # TODO: change name?
    """ Returns the resultant matrix of subtracting the given mean_face to each row in the given training_set.
    Note: The training_set and the mean_face must be represented as ndarrays.

    Params:
        training_set (ndarray): The training set represented as an ndarray where each row is an image.
        mean_face (ndarray): An ndarray representing the mean face.
    Returns:
         ndarray: The resulting matrix.
    """
    if not isinstance(training_set, np.ndarray):
        raise ValueError("Not an ndarray")
    mean_face = _mean_face(training_set)
    result = np.empty(training_set.shape)
    for row in range(0, training_set.shape[0]):
        result[row, :] = training_set[row, :] - mean_face
    # result = [training_set[row, :] - mean_face for row in range(training_set.shape[0])]
    return result


def _mean_face(training_set):
    """ Calculates the mean face of the training set of faces, given in matrix representation.

    Params:
        training_set (ndarray): The matrix representation of the training data set.
    Returns:
        ndarray: The mean face of the given training_set of faces (represented as an ndarray).
    """
    if not isinstance(training_set, np.ndarray):
        raise ValueError("Not an ndarray")
    return np.mean(training_set, 0)


def _l_matrix(difference_matrix):
    """ Calculates the L matrix.

    Params:
        difference_matrix (ndarray):
    Returns:
        matrix:
    """
    if not isinstance(difference_matrix, np.ndarray):
        raise ValueError("Not an ndarray")
    return np.matrix(difference_matrix) * np.matrix(difference_matrix.transpose())


def _separate_data_set(all_subjects, training_percentage):
    """ Separates the given data-set of subject's images into training and testing sets, in matrix representation.
    This methods expects the dictionary is a valid one.

    Params:
        all_subjects (dict): The dictionary holding each subject's list of images (represented as numpy's ndarray).
        training_percentage (float): The percentage of training elements.
        training_percentage (float): The percentage of training elements.
    Returns:
        dict: A dictionary with two keys: train and test.
        For each one, the values is the corresponding data-set splitted according the given training_percentage,
        in matrix representation
        (i.e each image, which in turn must be represented as an ndarray, is a row in the matrix).
    """
    if all_subjects is None or not isinstance(all_subjects, dict) or not all_subjects:
        raise ValueError("None, non-dictionary or empty subjects dictionary")
    matrix = _to_matrix_representation(all_subjects)  # TODO: agarrar mejor las cosas
    # We assume that it is a well formed images data set.
    training_amount = int(math.ceil(len(all_subjects) * len(all_subjects.values()[0]) * training_percentage))

    return {
        "train": matrix[0:training_amount, :],
        "test": matrix[training_amount:matrix.shape[0], :]
    }


def _to_matrix_representation(data_set):
    """ Transforms the given data set (i.e a dictionary with subject's images, represented as ndarrays) into a matrix,
    where each row is one image in the data set.

    Params:
        data_set (dict): The dictionary holding the images.
    Returns:
        ndarray: The matrix representation of the data-set.
    """
    if data_set is None or not isinstance(data_set, dict) or not data_set:
        raise ValueError("None, non-dictionary or empty subjects dictionary")
    image_size = data_set.values()[0][0].size
    list_of_matrices = map(lambda image_list: _list_to_matrix(image_list, image_size), data_set.values())

    return _list_to_matrix(list_of_matrices, image_size)


def _list_to_matrix(arrays_list, amount_of_columns=None):
    """ Transforms the given list of ndarrays into a matrix,
    appending those arrays into the resulting matrix at the bottom (i.e as a new row).
    Note: All ndarrays must have the same amount of columns.

    Params:
        arrays_list (list): The list holding the ndarrays.
        amount_of_columns (int, optional): The amount of columns of each ndarray.
        If not given, it is calculated from the first ndarray in the list.
    Returns:
        ndarray: The resulting matrix.
    """
    if amount_of_columns is not None and not isinstance(amount_of_columns, int):
        raise ValueError("The image size must be an int")
    if not arrays_list:
        raise ValueError("Can not transform an empty list into a matrix")
    if amount_of_columns is None:
        amount_of_columns = arrays_list[0].shape[1]  # We assume that the list is well formed

    matrix = np.zeros([0, amount_of_columns])
    for array in arrays_list:
        if array.shape[1] != amount_of_columns:
            raise ValueError("There were an ndarray with different size")
        matrix = np.append(matrix, array, axis=0)  # TODO: find a functional way to do this

    return matrix
