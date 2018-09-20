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
import pickle
import random
import copy
from win32api import GetKeyState
from win32con import VK_SCROLL, VK_CAPITAL

NORMALIZE_SIZE=150

###############################
# Run the program
#
def main( args ):
    from Utils.Training.config import Config
    print( "Initializing training...")
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    modelFile = args.outputFile

    print("Creating initial encodings...")
    initialEncodings = getEncodingsFromPaths( [args.imagePath], recursive=True, cache=True)

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
    morph2imageQueue = multiprocessing.Queue(maxsize=2*encBatchSize)
    image2encodingQueue = multiprocessing.Queue(maxsize=encBatchSize)
    encoding2morphQueue = multiprocessing.Queue()
    doneEvent = multiprocessing.Event()
    encodingDiedEvent = multiprocessing.Event()

    # Set up worker processes
    procs = []
    morphs2image = multiprocessing.Process(target=morphs_to_image_proc, args=( config,  morph2imageQueue, image2encodingQueue, doneEvent, args.pydev ) )
    procs.append(morphs2image)

    # Multiple encoding threads
    image2encoding = multiprocessing.Process(target=image_to_encoding_proc, args=( config, encBatchSize, image2encodingQueue, encoding2morphQueue, doneEvent, encodingDiedEvent, args.pydev ) )
    procs.append( image2encoding )

    neuralnet = multiprocessing.Process(target=neural_net_proc, args=( config, modelFile, trainBatchSize, initialEncodings, encoding2morphQueue, morph2imageQueue, doneEvent, args.pydev ) )
    procs.append(neuralnet)

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

    print("Enable ScrollLock to exit")
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
    for proc in procs:
        proc.join()


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
    encodedFaces = EncodedFace.batchEncode( imageList, batch_size=32, keepImage = True )

    return encodedFaces




def morphs_to_image_proc( config, inputQueue, outputQueue, doneEvent, pydev ):
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
            params = inputQueue.get(block=True, timeout=1)
            morphs = params[inputCnt:]
            vamFace.importFloatList(morphs)

            tmpdir = tempfile.mkdtemp( dir="D:/Generated/" )
            jsonFile = os.path.join( tmpdir, "face.json" )
            vamFace.save( jsonFile )
            vamWindow.loadLook( jsonFile, config.getAngles() )
            vamWindow.syncPipe( vamWindow._pipe )
            outputQueue.put( tmpdir )

        except queue.Empty:
            pass
        except Exception as e:
            print("Error in morphs_to_image_proc: {}".format(str(e)))


def image_to_encoding_proc( config, batchSize, inputQueue, outputQueue, doneEvent, encodingDiedEvent, pydev ):
    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    pathList = []
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    while not doneEvent.is_set():
        submitWork = False
        try:
            work = inputQueue.get(block=True, timeout=5)
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
                        outputQueue.put( ( params_valid, params ) )
                    except Exception as e:
                        # If encoding failed then feed a non-trainable random encoding to the output
                        #params = getRandomInputParams( config )
                        #params_valid = False
                        pass
                    #outputQueue.put( ( params_valid, params ) )
                    #shutil.rmtree( data[0], ignore_errors=True)
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
    pickle.dump( ( trainingInputs, trainingOutputs ), outFile )
    outFile.close()

def readTrainingData( dataName ):
    dataFile = open( dataName, "rb" )
    inputList, outputList = pickle.load( dataFile )
    return list(inputList), list(outputList)

def getRandomInputParams( config ):
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    params = list( np.random.normal(-1,1, inputCnt ) )
    params = params + [0]*outputCnt
    return params

def getRandomOutputParams( config, trainingMorphsList ):

    newFace = copy.deepcopy(config.getBaseFace())

    if len(trainingMorphsList) > 10:
        randomIdxs = random.sample( range(len(trainingMorphsList)), 2 )

        newFaceMorphs = trainingMorphsList[randomIdxs[0]]
        newFace.importFloatList( newFaceMorphs )

        # select which morphs to modify
        modifyIdxs = random.sample( range(len(newFace.morphFloats)), random.randint(1,50) )

        # How will we modify them?
        rand = random.random()
        if rand < .4:
            face2Morphs = trainingMorphsList[randomIdxs[1]]
            face2 = copy.deepcopy( config.getBaseFace() )
            face2.importFloatList( face2Morphs )
            mate(newFace, face2, modifyIdxs )
        elif rand < .9:
            for idx in modifyIdxs:
                newFace.changeMorph( idx, -1 + 2*random.random() )
        else:
            mutate(newFace, modifyIdxs )

    else:
        newFace.randomize()

    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    params = [0]*inputCnt
    params = params + newFace.morphFloats
    return params

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



def neural_net_proc( config, modelFile, batchSize, initialEncodings, inputQueue, outputQueue, doneEvent, pydev ):
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
    if os.path.exists( dataName ):
        trainingInputs,trainingOutputs = readTrainingData( dataName )
        print("Read {} training samples".format(len(trainingInputs)))
    batches = 0
    targetBatches = 0
    epochs=1
    pendingSave = False
    lastSave = time.time()

    while not doneEvent.is_set():
        try:
            valid, params = inputQueue.get(block=False, timeout=1)
            inputs = params[:inputCnt]
            outputs = params[inputCnt:]

            # If valid we can train on it
            if valid:
                trainingInputs.append( inputs )
                trainingOutputs.append( outputs )
                if time.time() > lastSave + 10*60 and len(trainingInputs) % 50 == 0:
                    pendingSave = True
                targetBatches += 1
                #print( "{} valid faces".format( len(trainingInputs) ) )

            if len(trainingInputs) > 0 and len(trainingInputs) % 100 == 0:
               # Periodically re-enqueue the initial encodings
                for encoding in initialEncodings:
                    try:
                        params = config.generateParams(encoding)
                        inputQueue.put( ( False, params ) )
                    except:
                        pass

            # Don't use predictions until we have trained a bit
            if len(trainingInputs) > 10000 and valid:
                # Now given the encoding, what morphs would we have predicted?
                predictedOutputs = create_prediction( neuralNet, np.array([inputs]) )
                predictedParams = inputs + list(predictedOutputs[0])
                outputQueue.put( predictedParams )

        except queue.Empty:
            if True or ( batches < targetBatches and len(trainingInputs) > 1000 ):
                batches += 1
                epochs = 1
                #if len(trainingInputs) > 25000:
                #    epochs = 5

                try:
                    while True:
                        outputQueue.put( getRandomOutputParams(config, trainingOutputs), block=False )
                except:
                    while True:
                        if len(trainingInputs) > 0:
                            neuralNet.fit( x=np.array(trainingInputs), y=np.array(trainingOutputs), batch_size=batchSize, epochs=epochs, verbose=1)
                        if not GetKeyState(VK_CAPITAL):
                            break



                if pendingSave:
                    print("Saving...")
                    neuralNet.save( modelFile )
                    saveTrainingData( dataName, trainingInputs, trainingOutputs)
                    print("Save complete!")
                    pendingSave = False
                    lastSave = time.time()


            else:
                try:
                    # Note: This block is what limits thread CPU usage
                    outputQueue.put( getRandomOutputParams(config, trainingOutputs), block=False, timeout=1 )
                except:
                    pass
                    # Bored? Add a random param
                    #inputQueue.put( ( False, getRandomInputParams(config) ) )


    print("Saving before exit...")
    neuralNet.save( modelFile )
    saveTrainingData( dataName, trainingInputs, trainingOutputs)
    print("Save complete.")

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

    model.add(Dense( 8*inputCnt + 2*outputCnt, input_shape=(inputCnt,), activation='linear', kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(3*inputCnt + 2*outputCnt, activation='linear', kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    model.add(Dense(3*inputCnt + 2*outputCnt, activation='linear', kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    model.add(Dense(3*inputCnt + 2*outputCnt, activation='linear', kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2*outputCnt, activation='linear', kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    model.add(Dense(outputCnt, activation='linear'))

    model.summary()

    input = Input(shape=(inputCnt,))
    predictor = model(input)
    nn = Model( input, predictor )

    optimizer = Adam(0.0002, 0.5)
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
    parser.add_argument('--imagePath', help="Root path for seed images", required=True)
    parser.add_argument('--seedJsonPath', help="Path to JSON looks to seed training with", default=None)
    parser.add_argument('--encBatchSize', help="Batch size for generating encodings", default=64)
    #parser.add_argument('--inputGlob', help="Glob for input images", default="D:/real/*.png")
    parser.add_argument('--outputFile', help="File to write output model to", default="output.model")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )