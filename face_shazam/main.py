# -*- coding: utf-8 -*-
import argparse
import logging
import sys

import image_utils
import face_recognition
from face_shazam import __version__

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Basic arguments
    parser = argparse.ArgumentParser(
        description="Face Shazam")
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version='face_shazam {ver}.'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="log_level",
        help="Set log level to INFO.",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="log_level",
        help="Set log level to DEBUG.",
        action='store_const',
        const=logging.DEBUG)

    # Image arguments
    parser.add_argument(
        "-s",
        "--subjects-path",
        dest="path_to_subjects",
        help="Set the path where the subjects images are. "
             "This path must contain subdirectories where each one has the images of a given subject.",
        default="./subjects",
        action="store",
        type=str
    )
    parser.add_argument(
        "-ext",
        "--image-extension",
        dest="image_extension",
        help="Set the file extension to be used for images.",
        default="pgm",
        action="store",
        type=str
    )

    # Training arguments
    parser.add_argument(
        "-tp",
        "--training-percentage",
        dest="training_percentage",
        help="Set the percentage of training data per subject.",
        default=0.6,
        action="store",
        type=float
    )
    parser.add_argument(
        "-e",
        "--energy-percentage",
        dest="energy_percentage",
        help="Set the percentage of eigen faces to be used by the recognizer training process.",
        default=None,
        action="store",
        type=float
    )
    parser.add_argument(
        "-kpd",
        "--kernel-polynomial-degree",
        dest="kernel_polynomial_degree",
        help="Set the kernel polynomial degree to be used with Kernel PCA recognition.",
        default=2,
        action="store",
        type=int
    )

    # Method arguments
    parser.add_argument(
        "-r",
        "--recognize-method",
        dest="recognize_method",
        help="Sets the recognition method (i.e PCA or Kernel PCA). Options to be used are: pca or kpca",
        default="pca",
        action="store",
        type=str.lower
    )

    # Subjects arguments
    parser.add_argument(
        "-S",
        "--subject",
        dest="subject_to_recognize",
        help="Sets path to the subject to be recognized",
        default=None,
        action="store",
        type=str
    )

    return parser.parse_args(args)


def setup_logging(log_level):
    """Setup basic logging

    Args:
      log_level (int): minimum loglevel for emitting messages
    """
    logging.basicConfig(level=log_level,
                        stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def create_recognizer(args, all_subjects):
    """ Returns the corresponding recognizer according to the set program arguments.

    Args:
        :obj:`argparse.Namespace`: command line parameters namespace
    Returns:
        FaceRecognizer: The already set recognizer (not trained).
    """
    recognizer = args.recognize_method
    training_percentage = args.training_percentage
    energy_percentage = args.energy_percentage
    if recognizer == "pca":
        _logger.info("Using PCA for face recognition")
        if "-kpd" in sys.argv or "kernel_polynomial_degree" in sys.argv:
            _logger.warn("The kernel polynomial degree must be used with kpca recognition.")
        return face_recognition.PCARecognizer(all_subjects, training_percentage, energy_percentage)
    elif recognizer == "kpca":
        _logger.info("Using Kernel-PCA for face recognition")
        kernel_polynomial_degree = args.kernel_polynomial_degree
        return face_recognition.KPCARecognizer(all_subjects, training_percentage, energy_percentage,
                                               kernel_polynomial_degree)


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.log_level)
    _logger.info("Starting application...")

    try:
        all_subjects = image_utils.get_subjects_dictionary(args.path_to_subjects, args.image_extension)
    except ValueError as e:
        _logger.error("There were validation errors with the images. Exception message: {}".format(e))
        exit(1)
    except IOError as e:
        _logger.error("There were IO errors while loading the images. Exception message: {}".format(e))
        exit(1)

    # noinspection PyUnboundLocalVariable
    recognizer = create_recognizer(args, all_subjects)  # If all_subjects is not initialized, exit(1) was executed
    recognizer.train()
    print("The score achieved is {}%".format(recognizer.test() * 100))
    subject_image_path = args.subject_to_recognize
    if subject_image_path is not None:
        subject_image = image_utils.get_one_image(subject_image_path)
        s = recognizer.recognize(subject_image, pre_process=True)
        print("Subject {} was recognized".format(s[0]))
    return


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
