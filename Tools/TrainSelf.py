# Generate training data from existing faces
import argparse
import os
import glob
import multiprocessing
import queue
import numpy as np
import tempfile
import shutil
import time
import random
import copy
import msgpack
import gc
import tqdm
from win32api import GetKeyState
from win32con import VK_SCROLL, VK_CAPITAL

NORMALIZE_SIZE=150

###############################
# Run the program
#
def main( args ):
    from Utils.Training.config import Config
    print( "Initializing training...")
    while GetKeyState(VK_SCROLL):
        print("Please turn off scroll lock")
        time.sleep(1)
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    modelFile = args.outputFile
    trainingCacheFile = args.trainingDataCache
    tmpDir = args.tmpDir
    onlySeed = args.onlySeedImages


    # If not none, the neural net process will generate from the training cache
    nnTrainingCache = None
    if args.useTrainingDataCache:
        nnTrainingCache = trainingCacheFile

    print("Creating initial encodings...")
    if args.seedImagePath is None:
        initialEncodings = []
    else:
        initialEncodings = getEncodingsFromPaths( [args.seedImagePath], recursive=True, cache=True)

    config = Config.createFromFile(args.configFile)
    # Try encodings until one succeeds
    initParams = None
    for encoding in initialEncodings:
        try:
            initParams = config.generateParams( encoding )
            break
        except:
            continue

    if initParams is None:
        raise Exception("Failed to create an initial encoding!")
    print("Shape is {}".format(config.getShape()))

    print("Starting child processes...")
    encBatchSize = args.encBatchSize
    trainBatchSize = 256
    morph2imageQueue = multiprocessing.Queue()
    image2encodingQueue = multiprocessing.Queue(maxsize=encBatchSize)
    encoding2morphQueue = multiprocessing.Queue()
    vamFaceQueue = multiprocessing.Queue()
    doneEvent = multiprocessing.Event()
    encodingDiedEvent = multiprocessing.Event()
    safeToExitEvents = []

    # Set up worker processes
    procs = []
    safeToExitEvent = multiprocessing.Event()
    morphs2image = multiprocessing.Process(target=morphs_to_image_proc, args=( config,  morph2imageQueue, image2encodingQueue, tmpDir, doneEvent, safeToExitEvent, args.pydev ) )
    procs.append(morphs2image)
    safeToExitEvents.append( safeToExitEvent )

    safeToExitEvent = multiprocessing.Event()
    image2encoding = multiprocessing.Process(target=image_to_encoding_proc, args=( config, encBatchSize, image2encodingQueue, encoding2morphQueue, vamFaceQueue, doneEvent, encodingDiedEvent, safeToExitEvent, args.pydev ) )
    procs.append( image2encoding )
    safeToExitEvents.append( safeToExitEvent )

    safeToExitEvent = multiprocessing.Event()
    neuralnet = multiprocessing.Process(target=neural_net_proc, args=( config, modelFile, trainBatchSize, initialEncodings, nnTrainingCache, encoding2morphQueue, morph2imageQueue, doneEvent, safeToExitEvent, onlySeed, args.pydev ) )
    procs.append(neuralnet)
    safeToExitEvents.append( safeToExitEvent )

    safeToExitEvent = multiprocessing.Event()
    trainingDataSaver = multiprocessing.Process( target=save_training_data_proc, args=( vamFaceQueue, trainingCacheFile, doneEvent, safeToExitEvent, args.pydev ))
    procs.append(trainingDataSaver)
    safeToExitEvents.append( safeToExitEvent )

    for proc in procs:
        proc.start()

    print("Begin processing!")

    #To kick start the process, feed the neural net the initial params
    for encoding in initialEncodings:
        try:
            params = config.generateParams(encoding)
            encoding2morphQueue.put( ( False, params ) )
        except:
            pass

    # Any seed json files?
    if args.seedJsonPath:
        seedLooks = getLooksFromPath( args.seedJsonPath )
        # Now match morphs and submit
        for look in seedLooks:
            look.matchMorphs( config.getBaseFace() )
            if len(look.morphFloats ) == config.getShape()[1]:
                morph2imageQueue.put( [0]*config.getShape()[0] + look.morphFloats )


    print("Enable ScrollLock to exit, CapsLock to pause image generation")
    while True:
        if GetKeyState(VK_SCROLL):
            break
        time.sleep(1)

        # image2encoding dies from OOM fairly often. Restart it if that happens
        if not image2encoding.is_alive() or encodingDiedEvent.is_set():
            print("Restarting Image2Encoding process!")
            encodingDiedEvent.clear()
            procs.remove( image2encoding )
            if image2encoding.is_alive():
                print("Terminating stuck process..")
                image2encoding.join(5)
                image2encoding.terminate()
            image2encoding = multiprocessing.Process(target=image_to_encoding_proc, args=( config, encBatchSize, image2encodingQueue, encoding2morphQueue, doneEvent, encodingDiedEvent, args.pydev ) )
            image2encoding.start()
            procs.append( image2encoding )

    print("Waiting for processes to exit...")
    # Wait for children to finish
    doneEvent.set()

#     for proc in procs:
#         proc.join()
    # Join isn't working right
    for exitEvent in safeToExitEvents:
        exitEvent.wait()

    print("Exit successful. If you're still stuck here, I don't know why. Just kill me.")

def getLooksFromPath( seedJsonPath, recursive = True ):
    from Utils.Face.vam import VamFace
    lookList = []
    for root, subdirs, files in os.walk(seedJsonPath):
        for file in files:
            if file.endswith( ( '.json'  ) ):
                try:
                    newFace = VamFace( os.path.join(root, file ) )
                    lookList.append(newFace)
                except:
                    pass
        if not recursive:
            break

    return lookList

def getEncodingsFromPaths( imagePaths, recursive = True, cache = False ):
    # We'll create a flat fileList, and placeholder arrays for the return encodings
    fileList = []
    encodings = []
    for imagePath in imagePaths:
        for root, subdirs, files in os.walk(imagePath):
            encoding = []
            for file in files:
                if file.endswith( ( '.png', '.jpg' ) ):
                    fileList.append( os.path.join( root, file ) )
                    encoding.append(None)
            if len(encoding) > 0:
                encodings.append(encoding)
            if not recursive:
                break

    # Now batch create the encodings!
    if len(fileList) > 0:
        batched_encodings = createEncodings( fileList )

    # Now unflatten the batched encodings
    idx = 0
    for encoding in encodings:
        for i in range(len(encoding)):
            encoding[i] = batched_encodings[idx]
            idx += 1

    return encodings


def createEncodings( fileList ):
    from PIL import Image
    from Utils.Face.encoded import EncodedFace

    imageList = []
    for file in fileList:
        imageList.append( np.array( Image.open(file) ) )
    encodedFaces = EncodedFace.batchEncode( imageList, batch_size=64, keepImage = True )

    return encodedFaces


# Previously we've just been saving the training number lists, but
# if we want to change a parameter we'd have to regenerate all data. This
# process saves the entire data set
def save_training_data_proc( inputQueue, trainingCacheFile, doneEvent, exitEvent, pydev ):
    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    trainingData = []

    if os.path.exists( trainingCacheFile ):
        trainingData = load_training_cache( trainingCacheFile )
        print("Loaded {} entries into training cache".format(len(trainingData)))


    saveInterval = 10000
    pendingSave = False

    while not doneEvent.is_set():
        try:
            faces = inputQueue.get(block=True, timeout=1)
            newEntry = []
            for face in faces:
                face._img = None # Don't want to save images
                newEntry.append(face)
            trainingData.append( newEntry )
            if len(trainingData) % saveInterval == 0:
                pendingSave = True
        except queue.Empty:
            if pendingSave:
                print("Appending {} entries to training cache...".format(len(trainingData)))
                save_training_cache( trainingData, trainingCacheFile )
                print("Done saving training cache")
                pendingSave = False
        except Exception as e:
            print("Error caching faces: {}".format(str(e)))

    print("Saving {} entries in training cache before exiting...".format(len(trainingData)))
    save_training_cache( trainingData, trainingCacheFile )
    print("Done saving training cache")
    exitEvent.set()

def load_training_cache( path ):
    from Utils.Face.encoded import EncodedFace
    import gc
    gc.disable()

    inFile = open( path, "rb")
    trainingData = msgpack.unpack( inFile, object_hook=EncodedFace.msgpack_decode, use_list=False, encoding='utf-8')
    inFile.close()

    gc.enable()
    return list(trainingData)

def save_training_cache( cacheData, path ):
    if len(cacheData) == 0:
        return

    from Utils.Face.encoded import EncodedFace
    outFile = open( path, 'wb' )
    msgpack.pack( cacheData, outFile, default=EncodedFace.msgpack_encode, use_bin_type=True)
    outFile.close()

def morphs_to_image_proc( config, inputQueue, outputQueue, tmpDir, doneEvent, exitEvent, pydev ):
    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    from Utils.Vam.window import VamWindow
    from Utils.Face.vam import VamFace
    # Initialize the Vam window
    vamWindow = VamWindow( pipe = "foto2vamPipe" )
    vamFace = config.getBaseFace()

    inputCnt = config.getShape()[0]


    while not doneEvent.is_set():
        try:
            while GetKeyState(VK_CAPITAL):
                time.sleep(1)
            params = inputQueue.get(block=True, timeout=1)
            morphs = params[inputCnt:]
            vamFace.importFloatList(morphs)

            tmpdir = tempfile.mkdtemp( dir=tmpDir )
            jsonFile = os.path.join( tmpdir, "face.json" )
            vamFace.save( jsonFile )
            vamWindow.loadLook( jsonFile, config.getAngles() )
            vamWindow.syncPipe( vamWindow._pipe )
            outputQueue.put( tmpdir )

        except queue.Empty:
            pass
        except Exception as e:
            print("Error in morphs_to_image_proc: {}".format(str(e)))

    exitEvent.set()


def image_to_encoding_proc( config, batchSize, inputQueue, outputQueue, trainingCacheQueue, doneEvent, encodingDiedEvent, exitEvent, pydev ):
    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    pathList = []
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    while not doneEvent.is_set():
        submitWork = False
        try:
            work = inputQueue.get(block=True, timeout=1)
            pathList.append( work )
            submitWork = len(pathList) >= batchSize
        except queue.Empty:
            submitWork = len(pathList) > 0

        if submitWork:
            try:
                encodings = getEncodingsFromPaths( pathList, recursive=False, cache = False )
                for data in zip( pathList, encodings ):
                    try:
                        if not validatePerson( data[1] ):
                            raise Exception("Image failed validation!")
                        params = config.generateParams( data[1] + [os.path.join( data[0], "face.json") ] )
                        params_valid = True
                        # Cache off the face
                        trainingCacheQueue.put( data[1] )
                        # Send it off to the neural net training
                        outputQueue.put( ( params_valid, params ) )
                    except Exception as e:
                        pass

            except RuntimeError as e:
                # Probably OOM. Kill the process
                print("RunTime error! Process is exiting")
                encodingDiedEvent.set()
                raise SystemExit()
            except Exception as e:
                print( str(e) )
            finally:
                for path in pathList:
                    try:
                        shutil.rmtree( path, ignore_errors=True)
                    except:
                        pass
                pathList.clear()

    exitEvent.set()

def validatePerson( encodingList ):
    ok = samePerson( encodingList, tolerance=0.6 )
    for encoding in encodingList:
        if not ok:
            break
        valid = landmarksValid( encoding )
        ok = valid > 0.9
    return ok


def samePerson( encodingList, tolerance=.6 ):
     for idx,encoding in enumerate(encodingList):
         for encoding2 in encodingList[idx+1:]:
             if encoding.compare(encoding2) > tolerance:
                 return False
     return True


def landmarksValid( encoding ):
    landmarks = encoding.getLandmarks()
    img = encoding.getImage()
    bgColor = img[0][0]

    totalPoints = 0
    invalidPoints = 0
    for feature,points in landmarks.items():
        for point in points:
            totalPoints += 1
            try:
                if (img[point[1]][point[0]] == bgColor).all():
                    invalidPoints += 1
            except IndexError:
                invalidPoints += 1
    return (totalPoints-invalidPoints)/totalPoints


def saveTrainingData( dataName, trainingInputs, trainingOutputs ):
    if len(trainingInputs) != len(trainingOutputs):
        raise Exception("Input length mismatch with output length!")

    outFile = open( dataName, 'wb' )
    msgpack.pack( ( trainingInputs, trainingOutputs ), outFile, use_bin_type=True )
    outFile.close()

def readTrainingData( dataName ):
    dataFile = open( dataName, "rb" )
    gc.disable()
    inputList,outputList = msgpack.unpack( dataFile )
    gc.enable()
    dataFile.close()
    return list(inputList), list(outputList)


def queueRandomOutputParams( config, trainingMorphsList, queue ):
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    inputParams = [0]*inputCnt

    newFace = copy.deepcopy(config.getBaseFace())

    # Choose random number to decide what modification we apply
    rand = random.random()
    # select which morphs to modify
    modifyIdxs = random.sample( range(len(newFace.morphFloats)), random.randint(1,25) )

    if len(trainingMorphsList) > 10:
        randomIdxs = random.sample( range(len(trainingMorphsList)), 2 )

        newFaceMorphs = trainingMorphsList[randomIdxs[0]]
        newFace.importFloatList( newFaceMorphs )

        if rand < .6:
            face2Morphs = trainingMorphsList[randomIdxs[1]]
            face2 = copy.deepcopy( config.getBaseFace() )
            face2.importFloatList( face2Morphs )
            mate(newFace, face2, modifyIdxs )
            queue.put_nowait( inputParams + newFace.morphFloats )
        elif rand < .9:
            for idx in modifyIdxs:
                newFace.changeMorph( idx, -1 + 2*random.random() )
                queue.put_nowait( inputParams + newFace.morphFloats )
        elif rand < .95:
            numSteps = 5#random.randint(5,15)
            for idx in modifyIdxs:
                face2 = copy.deepcopy( newFace )
                minVal = face2.morphInfo[idx]['min']
                maxVal = face2.morphInfo[idx]['max']
                stepSize = ( maxVal - minVal) / numSteps

                face2.morphFloats[idx] = minVal
                queue.put( inputParams + face2.morphFloats )
                for step in range(numSteps):
                    face2.changeMorph( idx, stepSize )
                    queue.put( inputParams + face2.morphFloats )
        else:
            mutate(newFace, modifyIdxs )
            queue.put_nowait( inputParams + newFace.morphFloats )
    else:
        # 90% chance to use baseface, otherwise completely random morphs.
        if rand < .9:
            mutate(newFace, modifyIdxs )
        else:
            newFace.randomize()
        queue.put_nowait( inputParams + newFace.morphFloats )


def mutate(face, idxList):
    for idx in idxList:
        face.randomize( idx )


def mate(targetFace, otherFace, idxList ):
    if len(targetFace.morphFloats) != len(otherFace.morphFloats):
        raise Exception("Morph float list didn't match! {} != {}".format(len(targetFace.morphFloats), len(otherFace.morphFloats)))
    for idx in idxList:
        weightA = random.randint(1,100)
        weightB = 100 - weightA
        matedValue = ( ( weightA * targetFace.morphFloats[idx] ) + ( weightB * otherFace.morphFloats[idx] ) ) / ( weightA + weightB )
        targetFace.morphFloats[idx] = matedValue



def neural_net_proc( config, modelFile, batchSize, initialEncodings, cacheToGenerateFrom, inputQueue, outputQueue, doneEvent, exitEvent, onlySeed, pydev ):
    # Work around low-memory GPU issue
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfConfig)

    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    dataName = modelFile + ".train"
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]

    trainingInputs = []
    trainingOutputs = []

    neuralNet = create_neural_net( inputCnt, outputCnt, modelFile )
    if os.path.exists( dataName ) and not onlySeed:
        trainingInputs,trainingOutputs = readTrainingData( dataName )

    lastSaveIdx = len(trainingInputs)
    pendingSave = False

    if cacheToGenerateFrom is not None:
        print("Currently have {} samples, now generating from training cache...".format(len(trainingInputs)))
        cache = load_training_cache( cacheToGenerateFrom )
        pendingSave = True
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for res in tqdm.tqdm(pool.imap_unordered( config.generateParams, cache ), total=len(cache) ):
            trainingInputs.append( res[:inputCnt] )
            trainingOutputs.append( res[inputCnt:] )
        pool.close()

#         start = time.time()
#         for idx,item in enumerate(cache):
#             try:
#                 newSample = config.generateParams(item)
#                 trainingInputs.append(newSample[:inputCnt])
#                 trainingOutputs.append(newSample[inputCnt:])
#                 if idx%500 == 0:
#                     print("Generating training data from cache {}/{}  [{}%]".format(idx,len(cache),round(100*(idx/len(cache)),2)))
#             except Exception as e:
#                 pass
#         print("Took {}s to generate from cache using 1 process".format( time.time() - start))

    print("Starting with {} training samples".format(len(trainingInputs)))

    lastSeedOnlyInputTime = 0
    lastSeedOnlyInputCount = 0
    lastSave = time.time()
    outputQueueSize = 256
    outputQueueSaveSize = 1024
    lastReEnqueueCnt = 0
    while not doneEvent.is_set():
        try:
            morphsValid, params = inputQueue.get(block=False)
            inputs = params[:inputCnt]
            outputs = params[inputCnt:]

            # If valid we can train on it
            if morphsValid:
                trainingInputs.append( inputs )
                trainingOutputs.append( outputs )
                if time.time() > lastSave + 120*60 and len(trainingInputs) % 50 == 0:
                    pendingSave = True
                #print( "{} valid faces".format( len(trainingInputs) ) )

            if len(trainingInputs) != lastReEnqueueCnt and len(trainingInputs) % 100 == 0:
                lastReEnqueueCnt = len(trainingInputs)
               # Periodically re-enqueue the initial encodings
                for encoding in initialEncodings:
                    try:
                        params = config.generateParams(encoding)
                        inputQueue.put( ( False, params ) )
                    except:
                        pass

            # Don't use predictions until we have trained a bit
            if ( len(trainingInputs) > 10000 ) or onlySeed:
                # Now given the encoding, what morphs would we have predicted?
                predictedOutputs = create_prediction( neuralNet, np.array([inputs]) )
                predictedParams = inputs + list(predictedOutputs[0])
                outputQueue.put( predictedParams )
                # Queue a random look for every predicted look. Sometimes we get stuck with
                # only predicted looks filling the queue, and it causes a downward spiral
                queueRandomOutputParams(config, trainingOutputs, outputQueue)

        except queue.Empty as e:
            # Been having issue with Queue Empty falsely triggering...
            if inputQueue.qsize() > 10:
                continue
            reqdSize = outputQueueSaveSize if pendingSave else outputQueueSize
            try:
                if not onlySeed:
                    while outputQueue.qsize() < reqdSize:
                        queueRandomOutputParams(config, trainingOutputs, outputQueue)
                elif ( len(trainingInputs) > lastSeedOnlyInputCount ) or ( time.time() > lastSeedOnlyInputTime + 10 ):
                    lastSeedOnlyInputCount = len(trainingInputs)
                    lastSeedOnlyInputTime = time.time()
                    for encoding in initialEncodings:
                        try:
                            params = config.generateParams(encoding)
                            inputQueue.put( ( False, params ) )
                        except:
                            pass

            finally:
                while True:
                    if len(trainingInputs) > 5000 or ( onlySeed and len(trainingInputs) > 0 ):
                        neuralNet.fit( x=np.array(trainingInputs), y=np.array(trainingOutputs), batch_size=batchSize, epochs=1, verbose=1)
                        pass
                    if not GetKeyState(VK_CAPITAL):
                        break
                    print("Caps Lock is enabled. Contiunally training and not feeding image generator")


            if pendingSave:
                print("Saving model...")
                neuralNet.save( modelFile )
                print("Done saving model, saving training data...")
                saveTrainingData( dataName, trainingInputs, trainingOutputs)
                lastSaveIdx = len(trainingInputs)
                print("Save complete!")
                lastSave = time.time()

            # Was our queue big enough to keep the generator busy while we trained?
            if outputQueue.qsize() == 0 and not onlySeed:
                if pendingSave:
                    outputQueueSize *= 1.5
                    print("Increased outputQueueSize to {}".format(outputQueueSize))
                else:
                    outputQueueSaveSize *= 1.15
                    print("Increased outputSaveQueueSize to {}".format(outputQueueSize))
            pendingSave = False


    print("Saving before exit...")
    neuralNet.save( modelFile )
    print("Model saved. Saving training data")
    saveTrainingData( dataName, trainingInputs, trainingOutputs)
    print("Save complete.")
    exitEvent.set()

def create_prediction( nn, input ):
    prediction = nn.predict(input)

    #limit range of output
    #prediction = np.around(np.clip(prediction, -1.5, 1.5 ),3)
    return prediction


def create_neural_net( inputCnt, outputCnt, modelPath ):
    from keras.models import load_model, Model, Sequential
    from keras.optimizers import Adam
    from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization

    if os.path.exists(modelPath):
        print("Loading existing model")
        return load_model(modelPath)

    model = Sequential()

    model.add(Dense(7*inputCnt, input_shape=(inputCnt,), kernel_initializer='random_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.3))

    model.add(Dense(7*inputCnt, kernel_initializer='random_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.3))

    model.add(Dense(7*inputCnt, kernel_initializer='random_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.3))

    model.add(Dense(7*inputCnt, kernel_initializer='random_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.3))

    model.add(Dense(outputCnt, activation='linear'))

    model.summary()

    input = Input(shape=(inputCnt,))
    predictor = model(input)
    nn = Model( input, predictor )

    optimizer = Adam(0.0002)
    nn.compile(loss='logcosh',
            optimizer=optimizer,
            metrics=['accuracy'])

    return nn

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Train GAN" )
    parser.add_argument('--configFile', help="Model configuration file", required=True)
    parser.add_argument('--seedImagePath', help="Root path for seed images. Must have at least 1 valid seed imageset", required=True)
    parser.add_argument('--onlySeedImages', action='store_true', default=False, help="Train *only* on the seed images")
    parser.add_argument('--seedJsonPath', help="Path to JSON looks to seed training with", default=None)
    parser.add_argument('--tmpDir', help="Directory to store temporary files. Recommend to use a RAM disk.", default='D:/Generated/')
    parser.add_argument('--encBatchSize', help="Batch size for generating encodings", default=64)
    parser.add_argument('--outputFile', help="File to write output model to", default="output.model")
    parser.add_argument('--trainingDataCache', help="File to cache raw training data", default="training.cache")
    parser.add_argument('--useTrainingDataCache', default=False, action='store_true', help="Generates training data from the cache and adds it to training data. Useful on first run with new config")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )