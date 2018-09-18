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
    encBatchSize = 16
    trainBatchSize = 512
    morph2imageQueue = multiprocessing.Queue()
    image2encodingQueue = multiprocessing.Queue(maxsize=encBatchSize)
    encoding2morphQueue = multiprocessing.Queue()
    doneEvent = multiprocessing.Event()


    # Set up worker processes
    procs = []
    morphs2image = multiprocessing.Process(target=morphs_to_image_proc, args=( config,  morph2imageQueue, image2encodingQueue, doneEvent, args.pydev ) )
    procs.append(morphs2image)

    # Multiple encoding threads
    image2encoding = multiprocessing.Process(target=image_to_encoding_proc, args=( config, encBatchSize, image2encodingQueue, encoding2morphQueue, doneEvent, args.pydev ) )
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

    input("Press Enter to exit...")

    # Wait for children to finish
    doneEvent.set()
    for proc in procs:
        proc.join()


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
            encodings.append(encoding)
            if not recursive:
                break

    # Now batch create the encodings!
    batched_encodings = createEncodings( fileList )

    # Now unflatten the batched encodings
    idx = 0
    for encoding in encodings:
        for i in range(len(encoding)):
            encoding[i] = batched_encodings[i]
            idx += 1

    return encodings


def createEncodings( fileList ):
    from PIL import Image
    from Utils.Face.encoded import EncodedFace

    imageList = []
    for file in fileList:
        imageList.append( np.array( Image.open(file) ) )
    encodedFaces = EncodedFace.batchEncode( imageList )

    #hack
    for face in encodedFaces:
        if face is None:
            continue

        if face.getAngle() < 0:
            face._angle= -face._angle
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


def image_to_encoding_proc( config, batchSize, inputQueue, outputQueue, doneEvent, pydev ):
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
                start = time.time()
                print("Submitting {} samples to GPU".format(len(pathList)))
                encodings = getEncodingsFromPaths( pathList, recursive=False, cache = False )
                for data in zip( pathList, encodings ):
                    try:
                        params = config.generateParams( data[1] + [os.path.join( data[0], "face.json") ] )
                        params_valid = True
                    except Exception as e:
                        # If encoding failed then feed a non-trainable random encoding to the output
                        params = getRandomInputParams( config )
                        params_valid = False
                    outputQueue.put( ( params_valid, params ) )
                    shutil.rmtree( data[0], ignore_errors=True)
                elapsed = time.time() - start
                print("Work done in {}  ({} per sample)!".format( elapsed, elapsed/len(pathList)))
                pathList.clear()
                batchSize *= 2
            except Exception as e:
                print( str(e) )
                



def saveTrainingData( dataName, trainingInputs, trainingOutputs ):
    if len(trainingInputs) != len(trainingOutputs):
        raise Exception("Input length mismatch with output length!")

    outFile = open( dataName, 'wb' )
    pickle.dump( ( trainingInputs, trainingOutputs ), outFile )
    outFile.close()

def readTrainingData( dataName ):
    dataFile = open( dataName, "rb" )
    inputList, outputList = pickle.load( dataFile )
    return inputList, outputList

def getRandomInputParams( config ):
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    params = list( np.random.normal(-1,1, inputCnt ) )
    params = params + [0]*outputCnt
    return params

def getRandomOutputParams( config ):
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    params = [0]*inputCnt
    params = params + list( np.random.normal(-1,1, outputCnt ) )
    return params


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
    lastSaved = 0
    while not doneEvent.is_set():
        try:
            valid, params = inputQueue.get(block=False, timeout=1)
            inputs = params[:inputCnt]
            outputs = params[inputCnt:]

            # If valid we can train on it
            if valid:
                trainingInputs.append( inputs )
                trainingOutputs.append( outputs )
                targetBatches += 1
                print( "{} valid faces".format( len(trainingInputs) ) )

            # Don't use predictions until we have trained a bit
            if len(trainingInputs) > 10000:
                # Now given the encoding, what morphs would we have predicted?
                predictedOutputs = create_prediction( neuralNet, np.array([inputs]) )
                predictedParams = inputs + list(predictedOutputs[0])
                outputQueue.put( predictedParams )

            outputQueue.put( getRandomOutputParams(config) )

        except queue.Empty:
            if batches < targetBatches and len(trainingInputs) > 1000:
                batches += 1
                epochs = 1
                if len(trainingInputs) > 25000:
                    epochs = 100

                neuralNet.fit( x=np.array(trainingInputs), y=np.array(trainingOutputs), batch_size=batchSize, epochs=epochs)
                #for i in range(epochs):
                    #neuralNet.train_on_batch( np.array(trainingInputs), np.array(trainingOutputs) )

                if len(trainingInputs) % 50 == 0 and lastSaved != len(trainingInputs):
                    print("Saving...")
                    neuralNet.save( modelFile )
                    saveTrainingData( dataName, trainingInputs, trainingOutputs)
                    lastSaved = len(trainingInputs)

                    # Periodically re-enqueue the initial encodings
                    for encoding in initialEncodings:
                        try:
                            params = config.generateParams(encoding)
                            inputQueue.put( ( False, params ) )
                        except:
                            pass


            else:
                try:
                    # Note: This block is what limits thread CPU usage
                    outputQueue.put( getRandomOutputParams(config), block=True, timeout=1 )
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
    prediction = np.around(np.clip(prediction, -1.5, 1.5 ),3)
    return prediction


def create_neural_net( inputCnt, outputCnt, modelPath ):
    from keras.models import load_model, Model, Sequential
    from keras.optimizers import Adam
    from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization

    if os.path.exists(modelPath):
        print("Loading existing model")
        return load_model(modelPath)

    model = Sequential()

    model.add(Dense( 8*inputCnt + 2*outputCnt, input_shape=(inputCnt,), kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(3*inputCnt + 2*outputCnt, kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2*outputCnt, kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))
    model.add(Dense(outputCnt, activation='linear'))

    model.summary()

    input = Input(shape=(inputCnt,))
    predictor = model(input)
    nn = Model( input, predictor )

    optimizer = Adam(0.0002, 0.5)
    nn.compile(loss='binary_crossentropy',
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