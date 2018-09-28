# Class to handle configurations

import json
import os
from Utils.Face.vam import VamFace
from Utils.Training.param_generator import ParamGenerator

class Config:
    CONFIG_VERSION = 1

    def __init__(self, configJson, basePath = "" ):
        minJson = os.path.join(basePath, configJson["minJson"]) if "minJson" in configJson else None
        maxJson = os.path.join(basePath, configJson["maxJson"]) if "maxJson" in configJson else None
        self._baseFace = VamFace( os.path.join(basePath, configJson["baseJson"]), minJson, maxJson )
        self._baseFace.trimToAnimatable()

        self._paramShape = None
        angles = set()
        self._input_params = []
        if "inputs" in configJson:
            inputs = configJson["inputs"]
            for param in inputs:
                try:
                    paramName = param["name"]
                    paramList = []
                    for paramParam in param["params"]:
                        paramList.append( { "name": paramParam["name"], "value": paramParam["value"] } )
                        if paramParam["name"] == "angle":
                            angles.add( float(paramParam["value"]))
                    self._input_params.append( { "name": paramName, "params": paramList } )
                except:
                    print("Error parsing parameter")


        self._output_params = []
        if "outputs" in configJson:
            outputs = configJson["outputs"]
            for param in outputs:
                try:
                    paramName = param["name"]
                    paramList = []
                    for paramParam in param["params"]:
                        paramList.append( { "name": paramParam["name"], "value": paramParam["value"] } )
                        if paramParam["name"] == "angle":
                            angles.add( float(paramParam["value"]))
                    self._output_params.append( { "name": paramName, "params": paramList } )
                except:
                    print("Error parsing parameter")


        self._angles = list(angles)
        self._angles.sort()

    @staticmethod
    def createFromFile( fileName ):
        data = open(fileName).read()
        jsonData = json.loads(data)

        if "config_version" in jsonData and jsonData["config_version"] is not Config.CONFIG_VERSION:
            raise Exception("Config version mismatch! File was {}, reader was {}".format(jsonData["config_version"], Config.CONFIG_VERSION ) )

        return Config( jsonData, os.path.dirname(fileName) )

    def getBaseFace(self):
        return self._baseFace

    def getShape(self):
        return self._paramShape

    def getAngles(self):
        return self._angles

    def generateParams(self, relatedFiles ):
        if self._paramShape is None:
            paramGen = ParamGenerator( self._input_params, self._angles, relatedFiles, self._baseFace )
            inputParams = paramGen.getParams()
            inputLen = len(inputParams)
            paramGen = ParamGenerator( self._output_params, self._angles, relatedFiles, self._baseFace )
            outputParams = paramGen.getParams()
            outputLen = len(outputParams)
            self._paramShape = (inputLen, outputLen)
            outParams = inputParams + outputParams
        else:
            paramGen = ParamGenerator( self._input_params + self._output_params, self._angles, relatedFiles, self._baseFace )
            outParams = paramGen.getParams()
        return outParams
