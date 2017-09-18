# -*- coding: utf-8 -*-
import itertools
import logging
import math

import numpy as np

_logger = logging.getLogger(__name__)


def process_faces(all_subjects, training_percentage):
    """ Processes the given data (i.e all_subjects).

    Params:
        all_subjects (dict): The dictionary holding each subject's list of images (represented as numpy's ndarray).
        training_percentage (float): The percentage of training elements.
    Returns:

    """
    # TODO complete returns
    _logger.info("Separating data set into training and testing data sets")
    separated_data_set = _separate_data_set(all_subjects, training_percentage)
    training_set = separated_data_set["train"][1]

    _logger.info("Normalizing the training data set in matrix representation")
    normalized_matrix = _get_normalized_matrix(training_set)

    _logger.info("Calculating the reduced covariance matrix")
    reduced_cov_matrix = normalized_matrix.transpose() * normalized_matrix

    _logger.info("Calculating eigen values and eigen vectors")
    eigen_values, eigen_vectors = np.linalg.eig(reduced_cov_matrix)  # TODO: Use own methods to calculate eigens

    _logger.info("Sorting the eigen vectors")
    sorted_indexes = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sorted_indexes]
    eigen_vectors = eigen_vectors[:, sorted_indexes]

    _logger.info("Calculating eigen faces")
    eigen_faces = normalized_matrix * eigen_vectors  # TODO: get only best ones

    _logger.info("Projecting training set in the eigen faces")
    projected_training_set = eigen_faces.transpose() * normalized_matrix

    return


def _get_normalized_matrix(training_set):
    """ Returns the resultant matrix of subtracting the mean by column
    to each column in the given training_set (in matrix representation).

    Params:
        training_set (matrix): The training set in matrix representation.
    Returns:
         matrix: The resulting matrix.
    """
    if not isinstance(training_set, np.matrix):
        raise ValueError("Not an ndarray")
    mean_face = np.mean(training_set, 1)  # Calculates the mean by column

    return training_set - mean_face


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
        in matrix representation, together with a list telling to which subject each column belongs to.
        (i.e each image, which in turn must be represented as an ndarray, is a column in the matrix).
    """
    if all_subjects is None or not isinstance(all_subjects, dict) or not all_subjects:
        raise ValueError("None, non-dictionary or empty subjects dictionary")

    # We assume data is well formed
    total_per_subject = len(all_subjects.values()[0])
    training_per_subject = int(math.ceil(total_per_subject * training_percentage))
    training_data = dict(itertools.izip(all_subjects.keys(),
                                        map(lambda image_list: image_list[0:training_per_subject],
                                            all_subjects.values())))
    testing_data = dict(itertools.izip(all_subjects.keys(),
                                       map(lambda image_list: image_list[training_per_subject:total_per_subject],
                                           all_subjects.values())))

    return {
        "train": _to_matrix_representation(training_data),
        "test": _to_matrix_representation(testing_data)
    }


def _to_matrix_representation(data_set):
    """ Transforms the given data set (i.e a dictionary with subject's images, represented as ndarrays) into a matrix,
    where each row is one image in the data set.

    Params:
        data_set (dict): The dictionary holding the images.
    Returns:
        tuple: A tuple holding subjects in each column
        in the matrix representation in the second element of the tuple.
    """
    if data_set is None or not isinstance(data_set, dict) or not data_set:
        raise ValueError("None, non-dictionary or empty subjects dictionary")
    image_shape = data_set.values()[0][0].shape
    subjects_matrices = map(lambda (subject, image_list): (subject, _list_to_matrix(image_list, image_shape)),
                            data_set.items())

    return _append_matrices(subjects_matrices)


def _list_to_matrix(arrays_list, original_shape=None):
    """ Transforms the given list of ndarrays into a matrix, first transforming each ndarray into a column ndarray
    (i.e amount of rows as the size of the array),
    and then appending those resulting arrays into the resulting matrix at the right (i.e as a new column).
    Note: All ndarrays must have the same shape.

    Params:
        arrays_list (list): The list holding the ndarrays.
    Returns:
        matrix: The resulting matrix.
    """
    if original_shape is not None and (not isinstance(original_shape, tuple) or not len(original_shape) == 2):
        raise ValueError("The original shape must be a tuple of two elements")
    if not arrays_list:
        raise ValueError("Can not transform an empty list into a matrix")
    if filter(lambda image_array: type(image_array) != np.ndarray, arrays_list):
        raise ValueError("There were elements in the list that are not specifically ndarrays")

    if original_shape is None:
        original_shape = arrays_list[0].shape

    amount_of_rows = original_shape[0] * original_shape[1]
    matrix = np.matrix(np.zeros([amount_of_rows, 0]))
    for array in arrays_list:
        if array.shape != original_shape:
            raise ValueError("There were an ndarray with different shape")
        array = np.reshape(array, [amount_of_rows, 1])
        matrix = np.append(matrix, array, axis=1)  # TODO: find a functional way to do this

    return matrix


def _append_matrices(subjects_matrices):
    """ Creates a tuple containing two elements where the first one is a dictionary holding subjects
    together with a list of numbers corresponding to columns in the matrix in the second element in the tuple.
    Those lists tell which column in the matrix belongs to each subject.

    Params:
        matrices_list (list): The list of tuples holding a subject and its matrix.
    Returns:
        tuple: A tuple holding subjects in each column in the matrix in the second element of the tuple.
    """
    # Validate types
    if not isinstance(subjects_matrices, list):
        raise ValueError("The subjects matrices must be a list")

    if filter(lambda tup: not isinstance(tup, tuple) or not isinstance(tup[0], str) or type(tup[1]) != np.matrix,
              subjects_matrices):
        raise ValueError("The list must contain tuples of two elements "
                         "where the first one is a string representing the subject "
                         "and the second one a matrix with the subject's data")

    # Check list is not empty
    if not subjects_matrices:
        raise ValueError("Can not transform an empty list into a matrix")

    amount_of_rows = subjects_matrices[0][1].shape[0]
    result_matrix = np.matrix(np.zeros([amount_of_rows, 0]))
    subjects = []
    for tup in subjects_matrices:
        if tup[1].shape[0] != amount_of_rows:
            raise ValueError("There were a matrix with different amount of rows")
        subjects += map(lambda each: tup[0], range(0, tup[1].shape[1]))
        result_matrix = np.append(result_matrix, tup[1], axis=1)

    return subjects, result_matrix
