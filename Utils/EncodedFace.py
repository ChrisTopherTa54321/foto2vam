# Class to handle faces encoded for recognition

import face_recognition
import numpy
from PIL import Image, ImageDraw


class EncodedFace:

    def __init__(self, image, region=None, keepImg=False, imgPadding=125):
        nImg = numpy.array(image)
        if region is None:
            self._region =  face_recognition.face_locations(nImg)[0]
            #print("Face found at {}".format(self._region))
        else:
            self._region = _region
        top,right,bottom,left = self._region

        # Apply padding to save more of image
        top = max(0, top-imgPadding)
        left = max(0, left-imgPadding)
        bottom = min(nImg.shape[0], bottom+imgPadding)
        right = min(nImg.shape[1], right+imgPadding)

        # crop image to just the face
        nImg = nImg[top:bottom, left:right]

        # Get encodings for the face in the image
        self._encodings = face_recognition.face_encodings(nImg)[0]

        # If we made it this far we have recognized the image. Save it for debugging?
        if keepImg:
            self._img = Image.fromarray(nImg)
            self._landmarks = face_recognition.face_landmarks(nImg)[0]
        else:
            self._img = None

    def getRegion(self):
        return self._region

    def compare(self, otherFace):
        return face_recognition.face_distance([self._encodings], otherFace._encodings).mean()

    def saveImage(self, filename, landmarks = True):
        if not self._img:
            raise Exception("Image was not saved in constructor!")

        img = self._img
        if landmarks:
            draw = ImageDraw.Draw(self._img)
            for key,val in self._landmarks.items():
                draw.point(val)

        img.save(filename)