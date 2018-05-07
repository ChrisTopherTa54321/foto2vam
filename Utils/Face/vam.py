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
        # reference to head rotation in the json
        self.headRotation = None

        # morphs as a list of floats
        self.morphFloats = []
        # valid ranges for each morph value
        self.morphInfo = []

        self.load( baseFileName )

        minFace = VamFace( minFileName ) if not minFileName is None else None
        maxFace = VamFace( maxFileName ) if not maxFileName is None else None

        # Create a list of floats representing each morph. Pull minimum and maximum
        # values, defaulting to 0-1.0 if a value is not present
        for morph in self.morphs:
            minVal = 0
            maxVal = 1.0
            defaultVal = 0

            val = minFace._getMorphValue(morph['name']) if not minFace is None else None
            minVal = float(val) if not val is None else minVal

            val = maxFace._getMorphValue(morph['name']) if not minFace is None else None
            maxVal = float(val) if not val is None else maxVal

            if 'value' in morph:
                defaultVal = float(morph['value'])
            self.morphFloats.append( defaultVal )
            self.morphInfo.append( { 'min': minVal, 'max': maxVal, 'name': morph['name'] } )


    # Load a JSON file
    def load(self, filename):
            data = open(filename).read()
            self.jsonData = json.loads(data)
            storables = self.jsonData["atoms"][0]["storables"]

            # Find the morphs in the json file
            storable = list(filter(lambda x : x['id'] == "geometry", storables))
            self.morphs = storable[0]['morphs']

            # Find the head rotation value in the json
            storable = list(filter(lambda x : x['id'] == "headControl", storables))
            if storable:
                self.headRotation = storable[0]['rotation']
            else:
                newNode = {
                          "id" : "headControl",
                          "position" : { "x" : 0, "y" : 0, "z": 0 },
                          "rotation" : { "x" : 0, "y" : 0, "z": 0 },
                          "positionState" : "On",
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
    def randomize(self):
        for idx in range(len(self.morphFloats)):
            self.morphFloats[idx] = random.uniform( self.morphInfo[idx]['min'], self.morphInfo[idx]['max'] )

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

    def _getMorphValue(self, key):
        morph = list( filter( lambda x : x['name'] == key, self.morphs ) )
        return morph[0]['value'] if len(morph) > 0 and 'value' in morph[0] else None