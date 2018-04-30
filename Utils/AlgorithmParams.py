# Encapsulate parameters to the algorithm
from . import EncodedFace
from win32api import GetKeyState
from win32con import VK_CAPITAL, VK_SCROLL
import random
import time
import os


###############################
class Params:
    
    
    ###############################
    #
    ###############################
    def __init__(self, protoFace, vamWindow, testJsonPath, targetFaces, outputPath, saveImages):
        print("Initialized algorithm. Since this simulates mouse clicks, your computer will not be usable during this process.")
        print("Turn on CAPS LOCK or SCROLL LOCK to suspend search to allow computer use.")
        self.protoFace = protoFace
        self.vamWindow = vamWindow
        self.jsonPath = testJsonPath
        self.targetFaces = targetFaces
        self.saveImages = saveImages
        self.outputPath = outputPath

        self.count = 0
        self.failures=0

        print( "Will be comparing faces to: ")
        sum = 0
        for angle,list in self.targetFaces.items():
            print( "{} faces at {} degree rotation".format( len(list), angle ))
            sum += len(list)
        print( "For a total of {} images to compare against and {} angles.".format(sum, len(self.targetFaces)))

    ###############################
    # Initialize an individual. Return the float-list representation of randomize morphs
    ###############################
    def initIndividual(self):
        # Randomize the face and return a new float list representing the morphs
        self.protoFace.randomize()
        return [] + self.protoFace.morphFloats

    ###############################
    # Return a fitness number, 0-1.0. Lower is better
    ###############################
    def evaluate(self, individual):
        # Give the user a chance to interrupt the process
        if (GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL)):
            print("WARNING: Suspending script due to Caps Lock or Scroll Lock being on. Push CTRL+PAUSE/BREAK or mash CTRL+C to exit script.")
            while GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL):
                time.sleep(1)

        # Default to 1.0 in case recognition fails
        face_distances = 1.0

        self.count += 1
        self.protoFace.importFloatList(individual)

        totalSum = 0
        failedAngles = []
        for angle,faces in self.targetFaces.items():
            self.protoFace.setRotation( angle )
            self.protoFace.save( self.jsonPath )
            self.vamWindow.loadLook()
            # Delay to allow time for the preset to load
            time.sleep(.2)

            image = self.vamWindow.getScreenShot()
        
            # Assume worst case distance
            distance = 1.0
            try:
                encodedFace = EncodedFace.EncodedFace(image, keepImg = self.saveImages)
                #print("Found face at {}".format(encodedFace.getRegion()))
                if self.saveImages:
                    encodedFace.saveImage(os.path.join(self.outputPath,'iteration_{}_angle_{}.png'.format(self.count, angle)))
                distanceSum = 0
                for idx,targetFace in enumerate(faces):
                    distanceSum += encodedFace.compare(targetFace)
                distance = distanceSum / len(faces)
            except:
                image.save("fails/failure{}.png".format(self.failures))
                self.failures += 1
                failedAngles.append(angle)

            totalSum += distance
            
        face_distance = totalSum / len(self.targetFaces)
        failureMsg = ""
        if len(failedAngles) > 0:
            failureMsg = "Failed to detect faces at these angles: {}".format(failedAngles)
        print("Comparison {} distance: {}\t{}".format(self.count, face_distance, failureMsg))
        return face_distance,

    ###############################
    # Randomly mutate an individual
    ###############################
    def mutate(self, individual, toolbox, mutProb ):
        # Create a new individual to mutate
        newInd = toolbox.clone(individual)
        #Now iterate through the float list, applying a mutation with probability mutProb
        for i in range(len(newInd)):
            if random.random() <= mutProb:
                newInd[i] += random.uniform(-0.25, .25)
        return newInd,

    ###############################
    # Not implemented
    ###############################
    def mate(self, ind1, ind2):
        print("Mate. Not implemented")
        return ind1, ind2

    ###############################
    # Ensure the individual is within min-max of face
    ###############################
    def checkBounds(self):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > self.protoFace.morphInfo[i]['max']:
                            child[i] = self.protoFace.morphInfo[i]['max']
                        elif child[i] < self.protoFace.morphInfo[i]['min']:
                            child[i] = self.protoFace.morphInfo[i]['min']
                return offspring
            return wrapper
        return decorator