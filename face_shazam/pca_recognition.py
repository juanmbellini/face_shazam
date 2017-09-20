# -*- coding: utf-8 -*-
import itertools
import logging
import math

import numpy as np
from sklearn.svm import LinearSVC

_logger = logging.getLogger(__name__)


class PCARecognizer:
    """ Class representing an face recognizer using PCA.
    """

    def __init__(self, data_set, training_percentage):
        """ Constructor.
        Params:
            data_set (dict): The dictionary holding each subject's list of images (represented as ndarrays).
            training_percentage (float): The percentage of training elements.
        """
        # TODO: will the recognizer receive training and testing data? or just training?
        # Validate data set
        if data_set is None or not isinstance(data_set, dict) or not data_set:
            raise ValueError("None, non-dictionary or empty subjects dictionary")
        if filter(lambda subject: not isinstance(subject, str), data_set.keys()):
            raise ValueError("The subjects must be represented as strings")
        if filter(lambda images: not isinstance(images, list) or filter(lambda elem: not isinstance(elem, np.ndarray),
                                                                        images), data_set.values()):
            raise ValueError("Each subject must have a list of ndarrays attached to it")
        images_per_subject = len(data_set.values()[0])
        if images_per_subject <= 0:
            raise ValueError("The data set must contain elements")
        images_shape = data_set.values()[0][0].shape
        if filter(lambda images: len(images) != images_per_subject or filter(lambda image: image.shape != images_shape,
                                                                             images), data_set.values()):
            raise ValueError("All subjects must have the same amount of images")
        # Validate training percentage
        if training_percentage is None or not isinstance(training_percentage, float) \
                or training_percentage <= 0.0 or training_percentage > 1.0:
            raise ValueError("None or out of range training percentage. Must be a float between 0 and 1")

        # Common values
        separated_data_set_container = self._SeparatedDataSetContainer(data_set, training_percentage)
        self._training_set = separated_data_set_container.training_set
        self._testing_set = separated_data_set_container.testing_set
        self._mean_face = np.mean(self._training_set.data, 1)  # Calculates the mean by column
        self._shape = images_shape

        # Values set after training
        self._eigen_faces = None
        self._clf = None
        self._trained = False

    def train(self, eigen_faces_amount):
        """ Trains this recognizer, using the given eigen_faces_amount.

        Params:
            eigen_faces_amount (int): The amount of eigen faces to be used.
        Returns:
            None
        """
        _logger.info("Normalizing the training data set in matrix representation")
        normalized_matrix = self._training_set.data - self._mean_face

        _logger.info("Calculating the reduced covariance matrix")
        reduced_cov_matrix = normalized_matrix.transpose() * normalized_matrix

        _logger.info("Calculating eigen values and eigen vectors")
        eigen_values, eigen_vectors = np.linalg.eig(reduced_cov_matrix)  # TODO: Use own methods to calculate eigens

        _logger.info("Sorting the eigen vectors")
        sorted_indexes = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sorted_indexes]
        eigen_vectors = eigen_vectors[:, sorted_indexes]

        _logger.info("Calculating eigen faces")
        self._eigen_faces = normalized_matrix * eigen_vectors
        self._eigen_faces = self._eigen_faces[:, range(0, eigen_faces_amount)]

        _logger.info("Projecting training set in the eigen faces")
        projected_training_set = self._eigen_faces.transpose() * normalized_matrix

        _logger.info("Training AI")
        self._clf = LinearSVC()
        self._clf.fit(projected_training_set.transpose(), self._training_set.subjects)
        self._trained = True

        return

    def test(self):
        """ Performs testing over this recognizer training.

        Returns:
            float: The score achieved by the testing process.
        """
        _logger.info("Testing AI")
        normalized_testing_set = self._testing_set.data - self._mean_face
        projected_testing_set = self._eigen_faces.transpose() * normalized_testing_set
        return self._clf.score(projected_testing_set.transpose(), self._testing_set.subjects)

    def recognize(self, image):
        """ Recognizes the given given picture, telling to whom it belongs.

        Params:
            picture (ndarray): The picture to be recognized.
            Must be an ndarray with the same shape of the images used to train.
        Returns:
            str: The string representation of the subject used to train the recognizer.l
        """
        # Validate state
        if not self._trained:
            raise ValueError("The recognizer must be trained before using this feature")
        # Validate types and values
        if not isinstance(image, np.ndarray):
            raise ValueError("The image to be recognized must be represented as an ndarray")
        if not image.shape == self._shape:
            raise ValueError("Can't recognize an image of different size than the used to train the recognizer")

        _logger.info("Trying to recognize the given picture")
        features = self._shape[0] * self._shape[1]
        return self._clf.predict(np.reshape(image, [1, features]))  # TODO: check what this returns

    class _SeparatedDataSetContainer:
        """ Class that wraps a training data set container and a testing data set container.
        """

        def __init__(self, data_set, training_percentage):
            """ Constructor.

            Params:
                data_set (dict): The dictionary holding each subject's list of images (represented as ndarrays).
                training_percentage (float): The percentage of training elements.
            """
            self._training_set, self._testing_set = self._separate_data_set(data_set, training_percentage)

        @property
        def training_set(self):
            """
            Returns:
                DataSetContainer: The data set container with training data.

            """
            return self._training_set

        @property
        def testing_set(self):
            """
            Returns:
                 DataSetContainer: The data set container with testing data.
            """
            return self._testing_set

        @classmethod
        def _separate_data_set(cls, all_subjects, training_percentage):
            """ Separates the given data-set of subject's images into training and testing sets,
            in a tuple of data set containers.

            Params:
                all_subjects (dict): The dictionary holding each subject's list of images (represented as ndarrays).
                training_percentage (float): The percentage of training elements.
            Returns:
                tuple: A tuple of two elements, holding the training data set container in the first one,
                and the testing data set in the second one.
            """
            # Arguments are already validated before (this is an inner class method called from constructor only)
            total_per_subject = len(all_subjects.values()[0])  # We assume data is well formed
            training_per_subject = int(math.ceil(total_per_subject * training_percentage))
            training_data = dict(itertools.izip(all_subjects.keys(),
                                                map(lambda image_list: image_list[0:training_per_subject],
                                                    all_subjects.values())))
            testing_data = dict(itertools.izip(all_subjects.keys(),
                                               map(lambda image_list: image_list[
                                                                      training_per_subject:total_per_subject],
                                                   all_subjects.values())))

            return cls.DataSetContainer(training_data), cls.DataSetContainer(testing_data)

        class DataSetContainer:
            """ Class that wraps the matrix representation of data (i.e images), together with a list of subjects.
            The list of subjects holds a representation of them in the same order as the data in the matrix.
            For example, if the first column of the matrix represents an image of "subject 1",
            the first element in the subjects list would be the representation of subject 1.
            Another example, if the n-th column in the matrix represents an image of "subject m",
            the n-th elements in the subjects list would be the representation of subject m.
            Note that a subject can be contained several times in the list, as more than one column in the matrix
            could represent an image of the said subject.
            """

            def __init__(self, data_set):
                """ Constructor.

                Params:
                    data_set (dict): The dictionary holding the subjects and their images.
                """
                self._subjects, self._data = self._to_matrix_representation(data_set)

            @property
            def subjects(self):
                """
                Returns:
                     list: A list of subject's representations.
                """
                return self._subjects

            @property
            def data(self):
                """
                Returns:
                    matrix: The matrix representation of data.
                """
                return self._data

            @classmethod
            def _to_matrix_representation(cls, data_set):
                """ Transforms the given data set (i.e a dictionary with subject's images, represented as ndarrays)
                into a matrix, where each row is one image in the data set.

                Params:
                    data_set (dict): The dictionary holding the images.
                Returns:
                    tuple: A tuple holding subjects in each column
                    in the matrix representation in the second element of the tuple.
                """
                if data_set is None or not isinstance(data_set, dict) or not data_set:
                    raise ValueError("None, non-dictionary or empty subjects dictionary")
                if filter(lambda subject: not isinstance(subject, str), data_set.keys()):
                    raise ValueError("The subjects must be represented as strings")
                if filter(lambda images: not isinstance(images, list) or filter(
                        lambda elem: not isinstance(elem, np.ndarray),
                        images), data_set.values()):
                    raise ValueError("Each subject must have a list of ndarrays attached to it")
                images_per_subject = len(data_set.values()[0])
                if images_per_subject <= 0:
                    raise ValueError("The data set must contain elements")
                images_shape = data_set.values()[0][0].shape
                if filter(lambda images: len(images) != images_per_subject or filter(
                        lambda image: image.shape != images_shape,
                        images), data_set.values()):
                    raise ValueError("All subjects must have the same amount of images")

                subjects_matrices = map(lambda (subject, image_list):
                                        (subject, cls._list_to_matrix(image_list, images_shape)), data_set.items())

                return cls._append_matrices(subjects_matrices)

            @staticmethod
            def _list_to_matrix(arrays_list, original_shape=None):
                """ Transforms the given list of ndarrays into a matrix,
                first transforming each ndarray into a column ndarray (i.e amount of rows as the size of the array),
                and then appending those resulting arrays into the resulting matrix at the right (i.e as a new column).
                Note: All ndarrays must have the same shape.

                Params:
                    arrays_list (list): The list holding the ndarrays.
                Returns:
                    matrix: The resulting matrix.
                """
                if original_shape is not None \
                        and (not isinstance(original_shape, tuple) or not len(original_shape) == 2):
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

            @staticmethod
            def _append_matrices(subjects_matrices):
                """ Creates a tuple containing two elements where the first one is a dictionary holding subjects
                together with a list of numbers corresponding to columns in the matrix
                in the second element in the tuple.
                Those lists tell which column in the matrix belongs to each subject.

                Params:
                    matrices_list (list): The list of tuples holding a subject and its matrix.
                Returns:
                    tuple: A tuple holding subjects in each column in the matrix in the second element of the tuple.
                """
                # Validate types
                if not isinstance(subjects_matrices, list):
                    raise ValueError("The subjects matrices must be a list")

                if filter(lambda t: not isinstance(t, tuple) or not isinstance(t[0], str) or type(t[1]) != np.matrix,
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
                    # subjects += map(lambda each: tup[0], range(0, tup[1].shape[1]))
                    subjects += [tup[0]] * tup[1].shape[1]
                    result_matrix = np.append(result_matrix, tup[1], axis=1)

                return subjects, result_matrix
