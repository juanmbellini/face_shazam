# -*- coding: utf-8 -*-
import functools
import itertools
import logging
import math
import abc

import numpy as np
from sklearn.svm import LinearSVC

_logger = logging.getLogger(__name__)


##############################
# Abstract recognizer
##############################

class FaceRecognizer:
    """ Class representing an abstract face recognizer
    """

    __metaclass__ = abc.ABCMeta  # Makes this class abstract

    @abc.abstractmethod
    def __init__(self, data_set, training_percentage, energy_percentage):
        """ Constructor.

        Params:
            data_set (dict): The dictionary holding each subject's list of images (represented as ndarrays).
            training_percentage (float): The percentage of training elements.
            energy_percentage (float): The percentage of energy (i.e sum of eigen values) to be used.
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
                or not (0 < training_percentage <= 1.0):
            raise ValueError("None, not float, or out of range training percentage. Must be a float between 0 and 1")
        # Validate eigen faces percentage
        if energy_percentage is not None \
                and (not isinstance(energy_percentage, float) or not (0 < energy_percentage <= 1.0)):
            raise ValueError("None float, or out of range eigen faces percentage. Must be a float between 0 and 1")

        # Common values
        separated_data_set_container = self._SeparatedDataSetContainer(data_set, training_percentage)
        self._training_set = separated_data_set_container.training_set
        self._testing_set = separated_data_set_container.testing_set
        self._shape = images_shape
        self._energy_percentage = energy_percentage

        # Values set after training
        self._clf = None
        self._trained = False

    ##############################
    # Interface methods
    ##############################

    def train(self):
        """ Trains this recognizer

        Returns:
             None
        """
        _logger.info("Training the recognizer. This process might take a while")
        training_array, classes_array = self._pre_train()

        _logger.info("Training AI")
        self._clf = LinearSVC()
        self._clf.fit(training_array, classes_array)
        self._trained = True
        return

    def test(self):
        """ Tests this recognizer.

        Returns:
            float: The score achieved by the testing process.
        """
        _logger.info("Testing the recognizer")
        testing_array, classes_array = self._pre_test()

        return self._clf.score(testing_array, classes_array)

    def recognize(self, image):
        """ Recognizes the given given picture, telling to whom it belongs.
        Note that if this recognizer was not trained with the subject to whom the given picture belongs to,
        it will classify it with any subject (will result in a false positive).

        Params:
            picture (ndarray): The picture to be recognized.
            Must be an ndarray with the same shape of the images used to train.
        Returns:
            str: The string representation of the subject used to train the recognizer.
        """
        if not self._trained:
            raise ValueError("The recognizer must be trained before using this feature")
        # Validate types and values
        if not isinstance(image, np.ndarray):
            raise ValueError("The image to be recognized must be represented as an ndarray")
        if not image.shape == self._shape:
            raise ValueError("Can't recognize an image of different size than the used to train the recognizer")

        _logger.info("Trying to recognize the given image")
        prediction_array = self._pre_recognize(image)
        return self._clf.predict(prediction_array)  # TODO: check what this returns

    ##############################
    # Abstract methods
    ##############################

    @abc.abstractmethod
    def _pre_train(self):
        """ Performs pre-training of the recognizer (i.e prepares the training and target arrays for the SVM).

        Returns:
            A tuple of two elements, where the first one is the training array, and the second one is the classes array.
            The training array must be a matrix which has each sample as a row, and each feature as a column.
            The classes array must be relative to the training array. For example, if sample n (in n-th column)
            belongs to class m, this list must contain m in the n-th position.
        """
        return None, None  # This discards "need more values to unpack" warning

    @abc.abstractmethod
    def _pre_test(self):
        """ Performs pre-testing operations, resulting in the testing and target arrays for the SVM

        Returns:
            A tuple of two elements, where the first one is the testing array, and the second one is the classes array.
            The testing array must be a matrix which has each sample as a row, and each feature as a column.
            The classes array must be relative to the testing array. For example, if sample n (in n-th column)
            belongs to class m, this list must contain m in the n-th position.
        """
        return None, None  # This discards "need more values to unpack" warning

    @abc.abstractmethod
    def _pre_recognize(self, image):
        """ Performs pre-recognition operations, resulting in the sample array to predict its class.

        Params:
            ndarray: The image in its ndarray representation.
        Returns:
            ndarray: The processed picture to recognize.
        """
        pass

    ##############################
    # Static methods
    ##############################

    @staticmethod
    def _calculate_eigens(matrix_, energy_percentage):
        """ Calculates the eigen values and eigen vectors of the given matrix_, sorting them in descending order
        according to the eigen values.
        If this recognizer has the _energy_percentage variable set, the amount of eigen values/vectors will be
        truncated according to that percentage (i.e that percentage is the percentage of the total sum of eigen values).

        Params:
            matrix_ (matrix): The matrix to which the eigen values/vectors must be calculated.
        Returns:
            tuple: A tuple of two elements containing the list of eigen values in the first element,
            and the eigen values in the second one, in matrix representation.
            Note that, if the list of eigen values is w, and the matrix of eigen vectors is w,
            eigen value v[i] corresponds to eigen vector w[:, i] .
        """
        _logger.info("Calculating eigen values and eigen vectors")
        eigen_values, eigen_vectors = np.linalg.eigh(matrix_)  # TODO: Use own methods to calculate eigens

        _logger.info("Sorting the eigen vectors")
        sorted_indexes = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sorted_indexes]
        eigen_vectors = eigen_vectors[:, sorted_indexes]

        # If percentage was set, get only those eigen vectors needed
        if energy_percentage is not None:
            _logger.info("Calculating total sum amount of eigen vectors needed")
            # Calculate total sum of eigen values
            eigen_values_sum = functools.reduce(lambda eigen1, eigen2: eigen1 + eigen2, eigen_values, 0.0)
            needed_sum = eigen_values_sum * energy_percentage
            # Calculate how many eigen vectors are needed to achieve the sum
            index_limit = 0
            actual_sum = 0.0
            while actual_sum < needed_sum:
                actual_sum += eigen_values[index_limit]
                index_limit += 1
            _logger.debug("{0} eigen vectors are needed to achieve an {1}% of the total sum"
                          .format(index_limit + 1, energy_percentage * 100))
            # Set results in eigen values list and eigen vectors matrix
            eigen_values = eigen_values[0:index_limit + 1]
            eigen_vectors = eigen_vectors[:, 0:index_limit + 1]
        return eigen_values, eigen_vectors

    ##############################
    # Helper classes
    ##############################

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


##############################
# Concrete recognizers
##############################

class PCARecognizer(FaceRecognizer):
    """ Class representing an face recognizer using PCA.
    """

    def __init__(self, data_set, training_percentage, eigen_faces_percentage):
        """ Constructor.
        Params:
            data_set (dict): The dictionary holding each subject's list of images (represented as ndarrays).
            training_percentage (float): The percentage of training elements.
            energy_percentage (float): The percentage of energy (i.e sum of eigen values) to be used.
        """
        # TODO: will the recognizer receive training and testing data? or just training?
        FaceRecognizer.__init__(self, data_set, training_percentage, eigen_faces_percentage)

        # Common values
        self._mean_face = np.mean(self._training_set.data, 1)  # Calculates the mean by column

        # Values set after training
        self._eigen_faces = None

    def _pre_train(self):
        """ Performs pre-training of the recognizer (i.e prepares the training and target arrays for the SVM).
        The pre-training process consist on:
            1. Subtract the mean face to the training set. We call this matrix the "normalized" matrix.
            2. Multiply the transpose of the the training set to the training set
                (resulting in the reduced covariance matrix)
            3. Calculate eigen vectors to the reduced covariance matrix
            4. Calculate eigen faces by multiplying the "normalized" matrix by each eigen vector got in the last step
                (see "Matthew Turk and Alex Pentland. Eigenfaces for recognition" (4) and (5) equations for more info.)
            5. Normalize the eigen faces (the last step maintains orthogonality, but not normality.
            6. Project the training set into the normalized eigen faces.
        Returns:
            A tuple of two elements, where the first one is the training array, and the second one is the classes array.
            The training array must be a matrix which has each sample as a row, and each feature as a column.
            The classes array must be relative to the training array. For example, if sample n (in n-th column)
            belongs to class m, this list must contain m in the n-th position.
        """
        _logger.info("Normalizing the training data set in matrix representation")
        normalized_matrix = self._training_set.data - self._mean_face

        _logger.info("Calculating the reduced covariance matrix")
        reduced_cov_matrix = normalized_matrix.transpose() * normalized_matrix

        eigen_values, eigen_vectors = self._calculate_eigens(reduced_cov_matrix, self._energy_percentage)
        _logger.info("Calculating eigen faces")
        self._eigen_faces = normalized_matrix * eigen_vectors

        # When calculating eigen faces, normalization is lost, so eigen faces won't be normalized
        # The following routine will normalize them, getting a better precision when training the SVM
        _logger.info("Normalizing eigen faces")
        for k in range(0, self._eigen_faces.shape[1]):
            eigen_face = self._eigen_faces[:, k]
            self._eigen_faces[:, k] = eigen_face / np.linalg.norm(eigen_face, ord=2)

        _logger.info("Projecting training set in {} eigen faces".format(self._eigen_faces.shape[1]))
        projected_training_set = self._eigen_faces.transpose() * normalized_matrix

        return projected_training_set.transpose(), self._training_set.subjects

    def _pre_test(self):
        """ Performs pre-testing operations, resulting in the testing and target arrays for the SVM
        The pre-testing process consist on:
            1. Subtract the mean face to the testing set. We call this matrix the "normalized" matrix.
            2. Project the "normalized" set into the normalized eigen faces.
        Returns:
            A tuple of two elements, where the first one is the testing array, and the second one is the classes array.
            The testing array must be a matrix which has each sample as a row, and each feature as a column.
            The classes array must be relative to the testing array. For example, if sample n (in n-th column)
            belongs to class m, this list must contain m in the n-th position.
        """
        normalized_testing_set = self._testing_set.data - self._mean_face
        projected_testing_set = self._eigen_faces.transpose() * normalized_testing_set
        return projected_testing_set.transpose(), self._testing_set.subjects

    def _pre_recognize(self, image):
        """ Performs pre-recognition operations, resulting in the sample array to predict its class.
        The pre-recognition process consist on:
            1. Reshape the image in order to be a matrix with the amount of rows as the amount of pixels it has,
                and one column.
            2. Subtract the mean face to the reshaped image. We call this matrix the "normalized" image.
            3. Project the "normalized" image into the normalized eigen faces.
        Params:
            ndarray: The image in its ndarray representation.
        Returns:
            ndarray: The processed picture to recognize.
        """
        _logger.info("Normalizing the image")
        features = self._shape[0] * self._shape[1]
        normalized_image = np.matrix(np.reshape(image, [features, 1])) - self._mean_face
        _logger.info("Projecting the image in the eigen faces")
        projected_image = self._eigen_faces.transpose() * normalized_image
        return projected_image.transpose()
