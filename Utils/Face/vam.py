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
        # reference to head rotation in the json
        self.headRotation = None
        # reference to storables in the json
        self._storables = None

        # morphs as a list of floats
        self.morphFloats = []
        # valid ranges for each morph value
        self.morphInfo = []

        # Abort out if not loading a file. Leaves face partially initialized
        if baseFileName is None:
            return

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

    # Note: msgpack really only is good for verifying a cache uses the same face, not for really saving off faces
    @staticmethod
    def msgpack_encode(obj):
        if isinstance(obj, VamFace):
            return {'__VamFace__': True, 'morphs': obj.morphs, 'minFace': obj.minFace, 'maxFace': obj.maxFace }
        return obj

    @staticmethod
    def msgpack_decode(obj):
        if '__VamFace__' in obj:
            decodedFace = VamFace(None)
            decodedFace.morphs = obj['morphs']
            decodedFace.minFace = obj['minFace']
            decodedFace.maxFace = obj['maxFace']
            decodedFace._createMorphFloats()
            obj = decodedFace
        return obj



    @staticmethod
    def mergeFaces( templateFace, fromFace, toFace, invertTemplate = False, copyNonMorphs = False ):
        newFace = copy.deepcopy( toFace )

        # Copy non-morphs, like clothes and skin
        if copyNonMorphs:
            for storable in fromFace._storables:
                id = storable['id'] if 'id' in storable else None
                # Storable must have an id, and we aren't currently copying morphs
                if id is None:
                    continue

                # Special case geometry, since we don't want to overwrite morphs
                if id == 'geometry':
                    newStorable = VamFace.getStorable( newFace._storables, id, create = True )
                    # Merge fromFace geometry with toFace
                    newStorable.update( storable )
                    # But keep toFace morphs for now
                    newStorable['morphs'] = VamFace.getStorable( toFace._storables, "geometry")["morphs"]
                else:
                    # Otherwise copy this morph into newFace
                    newStorable = VamFace.getStorable( newFace._storables, id, create = True )
                    newStorable.clear()
                    newStorable.update( storable )

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

            toMorph = toFace._getMorph( morph['name'] )

            # If we want this morph then copy it to newFace
            if copyMorph:
                morphCopy = morph.copy()
                # Maintain original animatable flag or clear it
                if toMorph and 'animatable' in toMorph:
                    morphCopy['animatable'] = toMorph['animatable']
                else:
                    morphCopy['animatable'] = False
                newMorphs.append( morphCopy )
                continue

            # Okay, we didn't want to copy it from fromFace, so keep the old morph (or set to 0)
            if not toMorph:
                morphCopy = morph.copy()
                morphCopy['value'] = 0
                morphCopy['animatable'] = False
            else:
                morphCopy = toMorph.copy()

            newMorphs.append( morphCopy )
#
        VamFace.setStorable( newFace._storables, "geometry", "morphs", newMorphs, create=True)
        newFace._createMorphFloats()

        return newFace

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

        if self._storables is not None:
            geometry = VamFace.getStorable( self._storables, "geometry", create=True )
            geometry['morphs'] = newMorphs
        self.morphs = newMorphs
        self._createMorphFloats()

    def trimToAnimatable(self):
        newMorphs = []
        print("Starting trim with {} morphs".format(len(self.morphs)))
        self.updateJson()
        for morph in self.morphs:
            if 'animatable' in morph:
                newMorphs.append(morph)
        geometry = VamFace.getStorable( self._storables, "geometry", create=True )
        geometry['morphs'] = newMorphs
        self.morphs = newMorphs
        print("Ending trim with {} morphs".format(len(self.morphs)))
        self._createMorphFloats()

    # Load a JSON file
    def load(self, filename, discardExtra = True ):
            data = open(filename).read()
            self.jsonData = json.loads(data)
            atoms = self.jsonData["atoms"][0]
            self._storables = atoms["storables"]

            # Get a reference to the object containing 'morphs' so we can completely replace 'morphs'
            geometry = VamFace.getStorable( self._storables, "geometry" )
            self.morphs = geometry['morphs']

            if discardExtra:
                # Check for male
                geometry = VamFace.getStorable( self._storables, "geometry")
                skin = "Female 1"
                if "character" in geometry and "Male" in geometry["character"]:
                    skin = "Male 1"

                # Throw away everything from storables
                self._storables = []
                atoms["storables"] = self._storables

                VamFace.setStorable( self._storables, "geometry", "morphs", self.morphs, create=True)
                VamFace.setStorable( self._storables, "geometry", "hair", "No Hair", create=True)
                VamFace.setStorable( self._storables, "geometry", "clothing", [], create=True)
                VamFace.setStorable( self._storables, "geometry", "character", skin, create=True)
                VamFace.setStorable( self._storables, "rescaleObject", "scale", 1.0 )
                VamFace.setStorable( self._storables, "JawControl", "targetRotationX", 0 )
                VamFace.setStorable( self._storables, "EyelidControl", "blinkEnabled", "false", create=True )
                VamFace.setStorable( self._storables, "AutoExpressions", "enabled", "false", create=True )

            # Find the head rotation value in the json
            VamFace.setStorable( self._storables, "headControl", "rotation", { "x": 0, "y": 0, "z": 0}, create=True )
            VamFace.setStorable( self._storables, "headControl", "positionState", "Off" )
            VamFace.setStorable( self._storables, "headControl", "rotationState", "On" )
            self.headRotation = VamFace.getStorable( self._storables, "headControl")['rotation']

    # Save json file
    def save(self, filename):
        self.updateJson()
        with open(filename, 'w') as outfile:
            json.dump(self.jsonData, outfile, indent=3)

    # randomize all face values
    def changeMorph(self, morphIdx, delta):
        newValue = self.morphFloats[morphIdx] + delta
        newValue = max( self.morphInfo[morphIdx]['min'], newValue )
        newValue = min( self.morphInfo[morphIdx]['max'], newValue )
        self.morphFloats[morphIdx] = newValue

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
                self.morphFloats[i] = float(floatList[i])
        else:
            raise Exception("Import list length [{}] is different than face's morph list length [{}]".format(len(floatList), len(self.morphFloats)))

    def setRotation(self, angle):
        self.headRotation['y'] = angle

    # Update the json with the values from the float list
    def updateJson(self, discardAnimatable = False):
        for idx,morph in enumerate(self.morphs):
            morph["value"] = self.morphFloats[idx]
            if discardAnimatable and 'animatable' in morph:
                del morph['animatable']


    def _getMorph(self, key):
        morph = list( filter( lambda x : x['name'] == key, self.morphs ) )
        if len(morph) > 0:
            return morph[0]
        return None

    def _getMorphValue(self, key):
        morph = self._getMorph(key)
        return morph['value'] if morph and 'value' in morph else None

    @staticmethod
    def setStorable(storables, id, param, value, create = False ):
        storable = VamFace.getStorable(storables, id, create)
        if storable:
            storable[param] = value
            return storable
        return None


    @staticmethod
    def getStorable( storables, id, create = False ):
        storable = list(filter(lambda x : x['id'] == id, storables ) )
        if len(storable) > 0:
            return storable[0]
        elif create:
            newNode = { "id": id }
            storables.append( newNode )
            return newNode
        return None
