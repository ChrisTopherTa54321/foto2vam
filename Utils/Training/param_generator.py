# Class to handle formatting data into csv parameters

import json
import csv
import math
from Utils.Face.encoded import EncodedFace
from Utils.Face.vam import VamFace

class ParamGenerator:

    def __init__(self, paramConfig, requiredAngles, relatedFiles, baseFace ):
        self._config = paramConfig
        self._encodings = []
        self._vamFaces = []
        self._angles = requiredAngles
        self._angles.sort()
        self._facebuckets = {}

        self._generators = { "encoding": self._encodingParams,
                             "json": self._jsonParams,
                             "eye_mouth_ratio": self._eye_mouth_ratio_params,
                             "mouth_chin_ratio": self._mouth_chin_ratio_params }

        for angle in self._angles:
            self._facebuckets[angle] = []

        # Read all encodings in from the file list
        for file in relatedFiles:
            try:
                newFace = EncodedFace.createFromFile(file)
                self._encodings.append(newFace)
            except:
                try:
                    vamFace = VamFace(file)
                    vamFace.matchMorphs( baseFace )
                    self._vamFaces.append( vamFace )
                except:
                    continue

        # Now put the encodings into the bucket with the closest angle
        for encoding in self._encodings:
            nearestBucket = abs(self._angles[0])
            for angle in self._angles:
                if abs( abs( encoding.getAngle() ) - abs( angle ) ) < abs( abs( encoding.getAngle() ) - abs(nearestBucket) ):
                    nearestBucket = abs(angle)
            self._facebuckets[nearestBucket].append(encoding)


    def getParams(self):
        outArray = []
        for param in self._config:
            if param["name"] in self._generators:
                outArray.extend(self._generators[param["name"]]( param["params"]))
        return outArray

    def _jsonParams(self, params):
        averages = None
        for face in self._vamFaces:
            if not averages:
                averages = [0] * len(face.morphFloats)
            for idx,val in enumerate(face.morphFloats):
                averages[idx] += val/len(self._vamFaces)
        return averages

    def _encodingParams(self, params):
        angle = None
        for param in params:
            if "name" in param and param["name"] == "angle":
                angle = float(param["value"])
                break

        encodings = []
        for encoding in self._facebuckets[angle]:
            encodings.append(encoding.getEncodings())

        averages = [0]*len(encodings[0])
        if len(encodings) > 0:
            for encoding in encodings:
                for idx,val in enumerate(encoding):
                    averages[idx] += val
            for idx,val in enumerate(averages):
                averages[idx] /= len(encodings)
        else:
            averages = encodings[0]

        return averages



    def _eye_mouth_ratio_params(self, params):
        angle = None
        for param in params:
            if "name" in param and param["name"] == "angle":
                angle = float(param["value"])
                break

        landmarks = []
        for encoding in self._facebuckets[angle]:
            landmarks.append(encoding.getLandmarks())

        averages = ParamGenerator._calcAverageSizes( landmarks )

        return [( averages["left_eye"][0] + averages["right_eye"][0] ) / (averages["top_lip"][0] + averages["bottom_lip"][0] )]



    def _mouth_chin_ratio_params(self, params):
        angle = None
        for param in params:
            if "name" in param and param["name"] == "angle":
                angle = float(param["value"])
                break

        landmarks = []
        for encoding in self._facebuckets[angle]:
            landmarks.append(encoding.getLandmarks())

        averages = ParamGenerator._calcAverageSizes( landmarks )

        return [averages["top_lip"][0] / averages["chin"][0]]



    @staticmethod
    def _calcAverageSizes( landmarks ):
        averages = {}
        for landmark in landmarks:
            sizes = ParamGenerator._calcSizes(landmark)
            for key,shape in sizes.items():
                if key not in averages:
                    averages[key] = [0,0]
                for idx,dim in enumerate(shape):
                    averages[key][idx] += ( dim / len(landmarks) )
        return averages

    @staticmethod
    def _calcSizes(landmarks):
        # Find the height and widths of all landmarks
        sizes = {}
        for key,pts in landmarks.items():
            leftmost = min( pts, key = lambda t: t[0] )
            rightmost = max( pts, key = lambda t: t[0] )
            highest = min( pts, key = lambda t: t[1] )
            lowest = max( pts, key = lambda t: t[1] )
            width = math.hypot( rightmost[0] - leftmost[0], rightmost[1] - leftmost[1] )
            height = math.hypot( lowest[0] - highest[0], lowest[1] - highest[1] )
            sizes[key] = [ width, height ]
        return sizes