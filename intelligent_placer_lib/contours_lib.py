import cv2
import mylogger
import math

_logger = mylogger.get_logger("logger")

class Contours:
    def __init__(self, polygon, objects, polygon_paper=None, objects_paper=None):
        self.polygon = polygon
        self.objects = objects
        self.polygon_paper = polygon_paper
        self.objects_paper = objects_paper

    @staticmethod
    def find_contours(image):
        # blur the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # highlight borders with Canny algorithm and save them as a mask
        canny_result = cv2.Canny(blurred, 50, 50 * 3, 2)
        mask = canny_result != 0
        image_tmp = 255 * (mask[:, :, None].astype(image.dtype))

        # Finding the contours of the borders
        all_contours, hierarchy = cv2.findContours(image_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Getting the contours of sheets of paper
        papers_indexes = [index for index in range(len(all_contours)) if hierarchy[0][index][3] < 0]
        if len(papers_indexes) < 2:
            _logger.error("Sheets of paper in the image are not recognized")
            return None

        # Leave only the largest outer contours
        if len(papers_indexes) > 2:
            papers_indexes.sort(key=lambda x: cv2.arcLength(all_contours[x], True))
            papers_indexes = [papers_indexes[-1], papers_indexes[-2]]

        # sort contours by coordinate y
        moments = [cv2.moments(all_contours[i]) for i in papers_indexes]
        y = [moment['m01']/moment['m00'] if moment['m00'] > 0 else math.inf for moment in moments]

        # save the contours of the top and bottom sheet
        if y[0] > y[1]:
            polygon_paper = all_contours[papers_indexes[1]]
            objects_paper = all_contours[papers_indexes[0]]
            papers_indexes = [papers_indexes[1], papers_indexes[0]]
        else:
            polygon_paper = all_contours[papers_indexes[0]]
            objects_paper = all_contours[papers_indexes[1]]
            papers_indexes = [papers_indexes[0], papers_indexes[1]]

        # find polygon
        polygon_indexes = [i for i in range(len(all_contours)) if hierarchy[0][hierarchy[0][i][3]][3] == papers_indexes[0]]
        if len(polygon_indexes) == 0:
            _logger.error("Could not find polygon")
            return None
        polygon = all_contours[max(polygon_indexes, key=lambda x: cv2.arcLength(all_contours[x], True))]
        if cv2.contourArea(polygon) < 1e-3:
            _logger.error("Polygon is not closed")
            return None

        objects_indexes = [i for i in range(len(all_contours)) if hierarchy[0][hierarchy[0][i][3]][3] == papers_indexes[1]]
        objects = [all_contours[index] for index in objects_indexes]
        if len(objects) == 0:
            return None

        # get rid of the inner lines of objects
        mask_image = image_tmp*0
        for o in objects:
            mask_image = cv2.drawContours(mask_image, o, -1, 255, thickness=2)
        mask_image = mask_image

        objects, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        external_indexes = [i for i in range(len(objects)) if hierarchy[0][i][3] < 0]

        objects = [objects[i] for i in external_indexes]

        return Contours(polygon, objects, polygon_paper, objects_paper)
