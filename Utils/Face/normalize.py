# Class to manipulate a normalize a face image
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import cv2
import dlib
import face_recognition_models
import numpy

class FaceNormalizer:

    def __init__(self, size=256, align = True, histogram = True):
        predictor = dlib.shape_predictor( face_recognition_models.pose_predictor_model_location() )
        self._detector = dlib.get_frontal_face_detector()
        self._size = size
        self._align = align
        self._histogram = histogram

        if self._align:
            self._aligner = FaceAligner( predictor=predictor, desiredFaceWidth = self._size)
        else:
            self._aligner = None


    def normalize(self, image):
        npImg = numpy.array(image)
        # PIL loads RGB, CV2 wants BGR
        npImg = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)
        npImg = imutils.resize(npImg, width=800)
        aligned = self._alignNpImg( npImg )
        if aligned is None:
            raise Exception("No face found in image!")
        npImg = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        return Image.fromarray(npImg)


    def _alignNpImg(self, npImg):
        gray = cv2.cvtColor(npImg, cv2.COLOR_BGR2GRAY)
        rects = self._detector(gray, 1)
        if len(rects) == 0:
            return None
        return self._aligner.align(npImg, gray, rects[0])