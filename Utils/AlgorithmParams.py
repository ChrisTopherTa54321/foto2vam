# Encapsulate parameters to the algorithm
from . import EncodedFace
from win32api import GetKeyState
from win32con import VK_CAPITAL, VK_SCROLL
import random
import multiprocessing
import time
import os
import queue


###############################
class Params:


    ###############################
    #
    ###############################
    def __init__(self, protoFace, vamWindow, testJsonPath, targetFaces, outputPath, numThreads, saveImages):
        print("Initialized algorithm. Since this simulates mouse clicks, your computer will not be usable during this process.")
        print("Turn on CAPS LOCK or SCROLL LOCK to suspend search to allow computer use.")
        self.protoFace = protoFace
        self.vamWindow = vamWindow
        self.jsonPath = testJsonPath
        self.targetFaces = targetFaces
        self.saveImages = saveImages
        self.outputPath = outputPath
        if numThreads > 1:
            self.poolWorkQueue = multiprocessing.Queue()
            self.poolResultQueue = multiprocessing.Queue()
            self.pool = []
            for idx in range(numThreads):
                proc = multiprocessing.Process(target=self.worker_process_func, args=(idx, self.poolWorkQueue, self.poolResultQueue) )
                proc.start()
                self.pool.append( proc )
        else:
            self.pool = None
        self.evalNum = 0
        self.failures=0
        self.bestCnt=0

        print( "Will be comparing faces to: ")
        imageCount = 0
        for angle,faceList in self.targetFaces.items():
            print( "{} faces at {} degree rotation".format( len(faceList), angle ))
            imageCount += len(faceList)
        print( "For a total of {} images to compare against and {} angles.".format(imageCount, len(self.targetFaces)))

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
        face_images = self.evaluate_get_face_images(individual)
        fitness = self.evaluate_get_fitness(face_images, self.evalNum)
        self.evalNum += 1

        return fitness

    ###############################
    # Return images of the test faces
    ###############################
    def evaluate_get_face_images(self, individual):
        # Give the user a chance to interrupt the process before we hijack the mouse
        if (GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL)):
            print("WARNING: Suspending script due to Caps Lock or Scroll Lock being on. Push CTRL+PAUSE/BREAK or mash CTRL+C to exit script.")
            while GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL):
                time.sleep(1)

        face_images = {}
        self.protoFace.importFloatList(individual)
        for angle,_ in self.targetFaces.items():
            self.protoFace.setRotation( angle )
            self.protoFace.save( self.jsonPath )
            self.vamWindow.loadLook()
            # Delay to allow time for the preset to load
            time.sleep(.3)
            face_images[angle] = self.vamWindow.getScreenShot()
        return face_images



    ###############################
    # Evaluate the fitness of the given images
    ###############################
    def evaluate_get_fitness(self, face_images, evalNum=0):
        # Default to 1.0 in case recognition fails
        totalSum = 0
        fitnesses = {}
        for angle,image in face_images.items():
            # Assume worst case distance in case of failure
            distance = 1.0
            try:
                targetFaces = self.targetFaces[angle]
                encodedFace = EncodedFace.EncodedFace(image, keepImg = self.saveImages)
                #print("Found face at {}".format(encodedFace.getRegion()))
                if self.saveImages:
                    encodedFace.saveImage(os.path.join(self.outputPath,'eval_{}_angle_{}.png'.format(evalNum, angle)))
                distanceSum = 0
                for targetFace in targetFaces:
                    distanceSum += encodedFace.compare(targetFace)
                distance = distanceSum / len(targetFaces)
            except:
                pass
                #image.save("fails/failure{}.png".format(self.failures))
                #self.failures += 1
            fitnesses[angle] = distance
            totalSum += distance

        face_distance = totalSum / len(self.targetFaces)
        print("Comparison {} fitness: {:.4f}\t{}".format(evalNum, face_distance, fitnesses))
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
                newInd[i] += random.uniform(-0.35, .35)
        return newInd,

    ###############################
    # Not implemented
    ###############################
    def mate(self, ind1, ind2):
        print("Mate. Not implemented")
        return ind1, ind2

    ###############################
    # Implement multithreading
    ###############################
    def map(self, partialFunc, params):
        # If no process pool then just call map directly
        if self.pool is None:
            ret = map(partialFunc, params)
        elif partialFunc.func == self.evaluate:
            ret = self.map_evaluate_function( params )
        else:
            raise NotImplementedError()
        return ret


    ###############################
    # Map the evaluate function over multiple processes
    ###############################
    def map_evaluate_function(self, params ):
        # On this thread we will screenshot faces, and will give the images to worker
        # processes in order to perform facial recognition on them
        expected_result_cnt = len(params)

        if not self.poolWorkQueue.empty() or not self.poolResultQueue.empty():
            raise Exception("Multiprocessing queues unexpectedly had data in them!")

        # Get images of the faces and submit them to the work queue
        for face in params:
            face_images = self.evaluate_get_face_images(face)
            self.poolWorkQueue.put( (self.evalNum, face_images) )
            self.evalNum += 1

        # Now wait for the worker processes to finish the work queue
        results = []
        while len(results) < expected_result_cnt:
            results.append( self.poolResultQueue.get(block=True) )

        # Verify nothing is weird with the queues
        if not self.poolWorkQueue.empty() or not self.poolResultQueue.empty():
            raise Exception("Multiprocessing queues unexpectedly had data in them!")

        # Now we have the results as tuples of (workId, fitness). Sort by workId and return fitness
        results.sort(key=lambda x: x[0])
        sorted_fitnesses = []
        for result in results:
            sorted_fitnesses.append(result[1])
        return sorted_fitnesses


    ###############################
    # Worker function for helper processes
    ###############################
    def worker_process_func(self, procId, workQueue, resultQueue):
        print("Worker {} started".format(procId))
        while True:
            try:
                work = workQueue.get(block=True, timeout=1)
                workId = work[0]
                workImages = work[1]
                fitness = self.evaluate_get_fitness(workImages, workId)
                resultQueue.put( (workId, fitness ) )
            except queue.Empty:
                pass
        print("Worker {} done!".format(procId))


    def saveBest(self, hof, args):
        self.bestCnt += 1
        self.protoFace.importFloatList(hof[0])
        self.protoFace.setRotation(0)
        self.protoFace.save(self.jsonPath)
        self.vamWindow.loadLook()

        # Delay to allow time for the preset to load
        time.sleep(.3)
        image = self.vamWindow.getScreenShot()

        encodedFace = EncodedFace.EncodedFace(image, keepImg = True)
        encodedFace.saveImage(os.path.join(self.outputPath,'best_{}.png'.format(self.bestCnt)))
        self.protoFace.save(os.path.join(self.outputPath, "best_{}.json".format(self.bestCnt)))
        return 0

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


    ###############################
    # Queue up the changed individual
    ###############################
    def registerChange(self):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                print("Change registered! {}".format(len(offspring)))
                for child in offspring:
                    print( child )
#                 for child in offspring:
#                     for i in range(len(child)):
#                         if child[i] > self.protoFace.morphInfo[i]['max']:
#                             child[i] = self.protoFace.morphInfo[i]['max']
#                         elif child[i] < self.protoFace.morphInfo[i]['min']:
#                             child[i] = self.protoFace.morphInfo[i]['min']
                return offspring
            return wrapper
        return decorator