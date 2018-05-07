# Class to manipulate a VAM face
import json
import random

class VamFace:
    wHndl = 0
    rect = ()

    # Initialize a base face from a JSON file
    # Get minimum and maximum values for parameters from minFace and maxFace files
    def __init__(self, baseFileName, minFileName = None, maxFileName = None):
        self.jsonData = {}

        # reference to the 'morphs' in the json
        self.morphs = None
        # reference to the parent of 'morphs' in the Json
        self.morphsContainer = None
        # reference to head rotation in the json
        self.headRotation = None

        # morphs as a list of floats
        self.morphFloats = []
        # valid ranges for each morph value
        self.morphInfo = []

        self.load( baseFileName )

        self.minFace = VamFace( minFileName ) if not minFileName is None else None
        self.maxFace = VamFace( maxFileName ) if not maxFileName is None else None

        self._createMorphFloats()

    def _createMorphFloats(self):
        # Create a list of floats representing each morph. Pull minimum and maximum
        # values, defaulting to 0-1.0 if a value is not present
        self.morphFloats = []
        self.morphInfo = []
        for morph in self.morphs:
            minVal = 0
            maxVal = 1.0
            defaultVal = 0

            val = self.minFace._getMorphValue(morph['name']) if not self.minFace is None else None
            minVal = float(val) if not val is None else minVal

            val = self.maxFace._getMorphValue(morph['name']) if not self.maxFace is None else None
            maxVal = float(val) if not val is None else maxVal

            if 'value' in morph:
                defaultVal = float(morph['value'])
            self.morphFloats.append( defaultVal )
            self.morphInfo.append( { 'min': minVal, 'max': maxVal, 'name': morph['name'] } )


    # Discard morphs in this face that are not in otherFace. Aligns morphFloats between the two faces.
    def matchMorphs(self, otherFace):
        newMorphs = []
        self.updateJson()

        # Keep only morphs that exist in the other face. If the other face has a morph that
        # we don't then add it as a 0 value
        for otherMorph in otherFace.morphs:
            morph = self._getMorph(otherMorph['name'])
            if not morph:
                # If we didn't have a value for the morph in the other face, then set to 0
                morph = otherMorph.copy()
                morph['value'] = 0
            newMorphs.append(morph)

        self.morphsContainer['morphs'] = newMorphs
        self.morphs = self.morphsContainer['morphs']
        self._createMorphFloats()

    def trimToAnimatable(self):
        newMorphs = []
        print("Starting trim with {} morphs".format(len(self.morphs)))
        self.updateJson()
        for morph in self.morphs:
            if 'animatable' in morph:
                newMorphs.append(morph)
        self.morphsContainer['morphs'] = newMorphs
        self.morphs = self.morphsContainer['morphs']
        print("Ending trim with {} morphs".format(len(self.morphs)))
        self._createMorphFloats()

    # Load a JSON file
    def load(self, filename):
            data = open(filename).read()
            self.jsonData = json.loads(data)
            storables = self.jsonData["atoms"][0]["storables"]

            # Find the morphs in the json file
            storable = list(filter(lambda x : x['id'] == "geometry", storables))

            # Dont' know how else to store a reference in Python
            self.morphsContainer = storable[0]
            self.morphs = self.morphsContainer['morphs']

            # Try to normalize the face pose
            self.morphsContainer["hair"] = "No Hair"
            self.morphsContainer["clothing"] = []
            self.morphsContainer["character"] = "Female 1"

            storable = list(filter(lambda x : x['id'] == "rescaleObject", storables))
            if len(storable) > 0:
                storable[0]["scale"] = 1.0

            storable = list(filter(lambda x : x['id'] == "EyelidControl", storables))
            if len(storable) > 0:
                storable[0]["blinkEnabled"] = "false"
            else:
                newNode = {
                          "id" : "EyelidControl",
                          "blinkEnabled" : "false"
                          }
                storables.append(newNode)

            storable = list(filter(lambda x : x['id'] == "AutoExpressions", storables))
            if len(storable) > 0:
                storable[0]["enabled"] = "false"
            else:
                newNode = {
                          "id" : "AutoExpressions",
                          "enabled" : "false"
                          }
                storables.append(newNode)

            storable = list(filter(lambda x : x['id'] == "JawControl", storables))
            if len(storable) > 0:
                storable[0]["targetRotationX"] = 0


            # Find the head rotation value in the json
            storable = list(filter(lambda x : x['id'] == "headControl", storables))
            if storable:
                storable[0]['rotation'] = { "x" : 0, "y" : 0, "z": 0 }
                self.headRotation = storable[0]['rotation']
            else:
                newNode = {
                          "id" : "headControl",
                          "rotation" : { "x" : 0, "y" : 0, "z": 0 },
                          "rotationState" : "On"
                          }
                storables.append(newNode)
                self.headRotation = newNode["rotation"]


    # Save json file
    def save(self, filename):
        self.updateJson()
        with open(filename, 'w') as outfile:
            json.dump(self.jsonData, outfile)


    # randomize all face values
    def randomize(self, morphIdx = None):
        if morphIdx is None:
            for idx in range(len(self.morphFloats)):
                self.morphFloats[idx] = random.uniform( self.morphInfo[idx]['min'], self.morphInfo[idx]['max'] )
        else:
            self.morphFloats[morphIdx] = random.uniform( self.morphInfo[morphIdx]['min'], self.morphInfo[morphIdx]['max'] )

    def importFloatList(self, floatList):
        if len(floatList) == len(self.morphFloats):
            for i in range(len(floatList)):
                self.morphFloats[i] = floatList[i]
        else:
            raise Exception("Import list length [{}] is different than face's morph list length [{}]".format(len(floatList), len(self.morphFloats)))

    def setRotation(self, angle):
        self.headRotation['y'] = angle

    # Update the json with the values from the float list
    def updateJson(self):
        for idx,morph in enumerate(self.morphs):
            morph["value"] = self.morphFloats[idx]

    def _getMorph(self, key):
        morph = list( filter( lambda x : x['name'] == key, self.morphs ) )
        if len(morph) > 0:
            return morph[0]
        return None

    def _getMorphValue(self, key):
        morph = self.getMorph(key)
        return morph['value'] if morph and 'value' in morph else None