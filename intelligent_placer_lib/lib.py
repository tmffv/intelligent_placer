import os.path
import cv2

from contours_lib import Contours
import mylogger

import matplotlib.pyplot as plt

_logger = mylogger.get_logger("logger")


def _check_path(path):
    if not os.path.exists(path) or not os.path.isfile(path) or not (path.endswith(".jpg") or path.endswith(".jpeg")):
        _logger.error("Incorrect path to image: %s" %path)
        return False
    return True


def _read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        _logger.error("failed to read image")
        return None
    return image


def draw_contours(contours, image):
    cv2.drawContours(image, contours.polygon_paper, -1, (255, 0, 0), thickness=3)
    cv2.drawContours(image, contours.objects_paper, -1, (0, 255, 0), thickness=3)
    cv2.drawContours(image, contours.polygon, -1, (255, 0, 255), thickness=3)
    cv2.drawContours(image, contours.objects, -1, (0, 0, 255), thickness=3)


def is_inside(image_path):
    if not _check_path(image_path):
        return False
    image = _read_image(image_path)
    if image is None:
        return False
    contours = Contours.find_contours(image)
    if contours is None:
        return False

    output_image = image.copy()
    draw_contours(contours, output_image)
    mask_image = image * 0 + 255
    draw_contours(contours, mask_image)
    cv2.imwrite("outputImages/%s" % image_path.split("/")[-1], output_image)
    cv2.imwrite("outputImages/mask_%s" % image_path.split("/")[-1], mask_image)

    return True
