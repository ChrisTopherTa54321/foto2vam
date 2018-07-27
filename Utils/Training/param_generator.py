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
                             "mouth_chin_ratio": self._mouth_chin_ratio_params,
                             "eye_height_width_ratio": self._eye_height_width_ratio_params,
                             "nose_height_width_ratio": self._nose_height_width_ratio_params,
                             "brow_height_width_ratio": self._brow_height_width_ratio_params,
                             "brow_chin_ratio": self._brow_chin_ratio_params,
                             "custom_action": self._custom_action  }

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
            else:
                raise Exception( "Generator {} not found!".format(param["name"]))
        return outArray

    def _jsonParams(self, params):
        averages = []
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

        if len(encodings) == 0:
            raise Exception( "No encodings found for angle {}".format(angle))
        averages = [0]*len(encodings[0])

        for encoding in encodings:
            for idx,val in enumerate(encoding):
                averages[idx] += val
        for idx,val in enumerate(averages):
            averages[idx] /= len(encodings)

        return averages



    def _eye_mouth_ratio_params(self, params):
        averages = self._getAverages( params )
        return [( averages["left_eye"][0] + averages["right_eye"][0] ) / (averages["top_lip"][0] + averages["bottom_lip"][0] )]

    def _eye_height_width_ratio_params(self, params):
        averages = self._getAverages( params )
        return [( averages["left_eye"][1] + averages["right_eye"][1] ) / ( averages["left_eye"][0] + averages["right_eye"][0] )]

    def _mouth_chin_ratio_params(self, params):
        averages = self._getAverages( params )
        return [averages["top_lip"][0] / averages["chin"][0]]


    def _nose_height_width_ratio_params(self, params):
        averages = self._getAverages( params )
        return [averages["nose_bridge"][1] / averages["nose_tip"][0]]


    def _brow_height_width_ratio_params(self, params):
        averages = self._getAverages( params )
        return [( averages["left_eyebrow"][1] + averages["right_eyebrow"][1] ) / ( averages["left_eyebrow"][0] + averages["right_eyebrow"][0] )]

    def _brow_chin_ratio_params(self, params):
        averages = self._getAverages( params )
        return [( averages["left_eyebrow"][0] + averages["right_eyebrow"][0] ) / ( averages["chin"][0]) ]


    @staticmethod
    def _vmAdd( workarea, param1, param2 ):
        l = ParamGenerator._vmResolveVariable( workarea, param1 )
        r = ParamGenerator._vmResolveVariable( workarea, param2 )
        return l+r;

    @staticmethod
    def _vmSub( workarea, param1, param2 ):
        l = ParamGenerator._vmResolveVariable( workarea, param1 )
        r = ParamGenerator._vmResolveVariable( workarea, param2 )
        return l-r;

    @staticmethod
    def _vmDiv( workarea, param1, param2 ):
        l = ParamGenerator._vmResolveVariable( workarea, param1 )
        r = ParamGenerator._vmResolveVariable( workarea, param2 )
        return l/r

    @staticmethod
    def _vmMult( workarea, param1, param2 ):
        l = ParamGenerator._vmResolveVariable( workarea, param1 )
        r = ParamGenerator._vmResolveVariable( workarea, param2 )
        return l*r;

    @staticmethod
    def _vmSet( workarea, param1, param2 ):
        return 0;

    @staticmethod
    def _vmResolveVariable( workarea, varName ):
        ret = None
        if '.' in varName:
            axisMap = { 'w': 0, 'h':1 }
            landmark,axis = varName.split('.')
            if landmark in workarea["landmarks"]:
                ret = workarea["landmarks"][landmark][axisMap[axis]]
        else:
            if varName in workarea["variables"]:
                ret = workarea["variables"][varName]
        return ret

    @staticmethod
    def _vmSetVariable( workarea, varName, value ):
        workarea["variables"][varName] = value

    def _custom_action(self, params):
        opcodes = {
                  "add": ParamGenerator._vmAdd,
                  "subtract": ParamGenerator._vmSub,
                  "divide": ParamGenerator._vmDiv,
                  "multiply": ParamGenerator._vmMult,
                  "set": ParamGenerator._vmSet
                   }

        averages = self._getAverages( params )
        actionArray = None
        for param in params:
            if param["name"] == "actions":
                actionArray = param["value"];

        if actionArray is not None:
            workArea = {}
            workArea["landmarks"] = averages
            workArea["variables"] = {}
            vm = {}
            for op in actionArray:
                opcode = op["op"]
                param1 = op["param1"] if "param1" in op else None
                param2 = op["param2"] if "param2" in op else None
                dest = op["dest"] if "dest" in op else None

                if opcode in opcodes:
                    opret = opcodes[opcode]( workArea, param1, param2 )
                    if dest:
                        ParamGenerator._vmSetVariable( workArea, dest, opret )
                elif opcode == "return":
                    return [ParamGenerator._vmResolveVariable(workArea, param1)]
        raise Exception( "Ill-formed action: {}".format(params))

    def _getAverages(self, params):
        angle = None
        for param in params:
            if "name" in param and param["name"] == "angle":
                angle = float(param["value"])
                break

        landmarks = []
        for encoding in self._facebuckets[angle]:
            landmarks.append(encoding.getLandmarks())

        averages = ParamGenerator._calcAverageSizes( landmarks )
        return averages


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