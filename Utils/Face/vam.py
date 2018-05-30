# Class to manipulate a VAM face
import json
import random
import copy

class VamFace:
    wHndl = 0
    rect = ()

    # Initialize a base face from a JSON file
    # Get minimum and maximum values for parameters from minFace and maxFace files
    def __init__(self, baseFileName, minFileName = None, maxFileName = None, discardExtra = True):
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

        self.load( baseFileName, discardExtra = discardExtra )

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


    @staticmethod
    def mergeFaces( templateFace, fromFace, toFace, invertTemplate = False, copyNonMorphs = False ):
        newFace = copy.deepcopy( toFace )

        # Copy non-morphs, like clothes and skin
        if copyNonMorphs:
            for node,val in fromFace.morphsContainer.items():
                if node != "morphs":
                    newFace.morphsContainer[node] = val

        # Now copy, based on the template, the morphs
        newMorphs = []
        for morph in fromFace.morphs:
            # First check the template to see if we want to copy the morph
            templateMorph = templateFace._getMorph( morph['name'] )

            copyMorph = False
            if templateMorph and "animatable" in templateMorph and templateMorph["animatable"]:
                copyMorph = True

            if invertTemplate:
                copyMorph = not copyMorph

            # If we want this morph then copy it to newFace
            if copyMorph:
                newMorphs.append( morph )
                continue

            # Okay, we didn't want to copy it from fromFace, so keep the old morph (or set to 0)
            oldMorph = toFace._getMorph( morph['name'] )
            if not oldMorph:
                oldMorph = morph.copy()
                oldMorph['value'] = 0
            newMorphs.append( oldMorph )
#
        newFace.morphsContainer['morphs'] = newMorphs
        newFace.morphs = newFace.morphsContainer['morphs']
        newFace._createMorphFloats()

        return newFace


    def copyNonMorphs(self, fromFace ):

        pass
#         toFace.matchMorphs(fromFace)
#         for morph in fromFace.morphs:
#             if fromFace.getMorph( morph['name'] ):
#                 if morph['name'] is in fromFace.morphs:
#
#
#
#         # Create a copy of 'fromFace' that only contains morphs to copy
#         scratchFromFace = copy.deepcopy( fromFace )
#         scratchFromFace.matchMorphs( templateFace )
#         scratchToFace = copy.deepcopy( toFace )
#
#         # ScratchFromFace contains only the values we want to copy
#         # scratchToFace contains the face to copy to.
#         # Now, overwrite scratchToFace parameters with ScratchFromFace params
#
#         for otherMorph in scratchFromFace.morphs:
#             morph = scratchToFace._getMorph(otherMorph['name'])
#             if not morph:
#                 scratchToFace.morphs.append( otherMorph )
#             else:
#                 morph = otherMorph
#
#         scratchToFace.morphsContainer['morphs'] = scratchToFace.morphs
#         scratchToFace.morphs = scratchToFace.morphsContainer['morphs']
#         scratchToFace._createMorphFloats()
#         return scratchToFace


    # Aligns morphFloats between the two faces, discarding any morphs not in otherFace
    def matchMorphs(self, otherFace, copyUnknowns = False, templateFace = None, invertTemplate = False):
        newMorphs = []
        self.updateJson()

        # Loop through other morph, copying any required morphs
        for otherMorph in otherFace.morphs:
            if templateFace:
                templateMorph = templateFace._getMorph( otherMorph['name'] )

                if templateMorph and "animatable" in templateMorph and templateMorph["animatable"]:
                    copyMorph = True

                if invertTemplate:
                    copyMorph = not copyMorph

                # If we don't want to copy this morph, just copy the original morph
                if not copyMorph:
                    newMorphs.append( otherMorph )
                    continue

            morph = self._getMorph(otherMorph['name'])

            if not morph:
                # If we didn't have a value for the morph in the other face, then set to 0
                morph = otherMorph.copy()
                if not copyUnknowns:
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
    def load(self, filename, discardExtra = True ):
            data = open(filename).read()
            self.jsonData = json.loads(data)
            storables = self.jsonData["atoms"][0]["storables"]

            # Find the morphs in the json file
            storable = list(filter(lambda x : x['id'] == "geometry", storables))

            # Dont' know how else to store a reference in Python
            self.morphsContainer = storable[0]
            self.morphs = self.morphsContainer['morphs']

            if discardExtra:
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