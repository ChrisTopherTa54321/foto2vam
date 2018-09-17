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
import csv

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
    initialEncodings = getEncodingsFromPath(args.imagePath, recursive=True, cache=True)

    config = Config.createFromFile(args.configFile)
    initParams = config.generateParams( initialEncodings[0] )
    print("Shape is {}".format(config.getShape()))

    print("Starting child processes...")
    morph2imageQueue = multiprocessing.Queue()
    image2encodingQueue = multiprocessing.Queue(maxsize=20)
    encoding2morphQueue = multiprocessing.Queue()
    doneEvent = multiprocessing.Event()

    # Set up worker processes
    procs = []
    morphs2image = multiprocessing.Process(target=morphs_to_image_proc, args=( config, morph2imageQueue, image2encodingQueue, doneEvent, args.pydev ) )
    procs.append(morphs2image)

    # Multiple encoding threads
    for idx in range(7):
        image2encoding = multiprocessing.Process(target=image_to_encoding_proc, args=( config, image2encodingQueue, encoding2morphQueue, doneEvent, args.pydev ) )
        procs.append( image2encoding )

    neuralnet = multiprocessing.Process(target=neural_net_proc, args=( config, modelFile, initialEncodings, encoding2morphQueue, morph2imageQueue, doneEvent, args.pydev ) )
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


def getEncodingsFromPath( imagePath, recursive = True, cache = False ):
    from Utils.Face.encoded import EncodedFace
    encodingList = []
    for root, subdirs, files in os.walk(imagePath):
        encoding = []
        for file in files:
            if file.endswith( ('.png', '.jpg')):
                try:
                    fileName = os.path.join(root,file)
                    encodingName = fileName + "_encoding"
                    if cache and os.path.exists(encodingName):
                        newEncoding = EncodedFace.createFromFile(encodingName)
                    else:
                        newEncoding = createEncoding( fileName )
                        if cache:
                            newEncoding.saveEncodings( encodingName )

                    encoding.append( newEncoding )
                except:
                    pass
        if len(encoding) > 0:
            encodingList.append(encoding)

        if not recursive:
            break
    return encodingList


def createEncoding( imageFile ):
    from PIL import Image
    from Utils.Face.normalize import FaceNormalizer
    from Utils.Face.encoded import EncodedFace
    img = Image.open(imageFile)
    # Supposedly dlib takes care of normalization
    # normalizer = FaceNormalizer(NORMALIZE_SIZE)
    #img = normalizer.normalize(img)
    encodedFace = EncodedFace( img, num_jitters=1 )
    return encodedFace




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


def image_to_encoding_proc( config, inputQueue, outputQueue, doneEvent, pydev ):
    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]
    while not doneEvent.is_set():
        try:
            work = inputQueue.get(block=True, timeout=1)
            try:
                encodings = getEncodingsFromPath( work )
                params = config.generateParams( encodings[0] + [os.path.join( work, "face.json" )] )
                params_valid = True
            except Exception as e:
                # If encoding failed then feed a non-trainable random encoding to the output
                params = getRandomInputParams( config )
                params_valid = False

            outputQueue.put( ( params_valid, params ) )
            shutil.rmtree( work, ignore_errors=True )
        except queue.Empty:
            pass


def saveCsv( csvName, trainingInputs, trainingOutputs ):
    if len(trainingInputs) != len(trainingOutputs):
        raise Exception("Input length mismatch with output length!")
    
    outFile = open( csvName, 'w' )
    writer = csv.writer( outFile, lineterminator='\n')
    writer.writerow( [ len(trainingInputs[0]), len(trainingOutputs[0])])
    for idx in range(len(trainingInputs)):
        writer.writerow( trainingInputs[idx] + trainingOutputs[idx] )
    outFile.close()

def readCsv( csvName ):
    inputList = []
    outputList = []
    with open(csvName, newline='\n') as f_input:
        contents = [list(map(float, row)) for row in csv.reader(f_input)]
    inputCnt,outputCnt = contents[0]
    inputCnt = int(inputCnt)
    outputCnt = int(outputCnt)
    for i in range(1, len(contents)):
        inputList.append( contents[i][:inputCnt] )
        outputList.append( contents[i][inputCnt:] )
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


def neural_net_proc( config, modelFile, initialEncodings, inputQueue, outputQueue, doneEvent, pydev ):
    # Work around low-memory GPU issue
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfConfig)

    if pydev:
        import pydevd
        pydevd.settrace(suspend=False)

    csvName = modelFile + ".csv"
    inputCnt = config.getShape()[0]
    outputCnt = config.getShape()[1]

    trainingInputs = []
    trainingOutputs = []

    neuralNet = create_neural_net( inputCnt, outputCnt, modelFile )
    if os.path.exists( csvName ):
        trainingInputs,trainingOutputs = readCsv( csvName )
        print("Read {} training samples".format(len(trainingInputs)))
    batches = 0
    targetBatches = 0
    lastSaved = 0
    while not doneEvent.is_set():
        try:
            valid, params = inputQueue.get(block=True, timeout=1)
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
                for i in range(epochs):
                    neuralNet.train_on_batch( np.array(trainingInputs), np.array(trainingOutputs) )

                if len(trainingInputs) % 50 == 0 and lastSaved != len(trainingInputs):
                    print("Saving...")
                    neuralNet.save( modelFile )
                    saveCsv( csvName, trainingInputs, trainingOutputs)
                    lastSaved = len(trainingInputs)

                    # Periodically re-enqueue the initial encodings
                    for encoding in initialEncodings:
                        try:
                            params = config.generateParams(encoding)
                            inputQueue.put( ( False, params ) )
                        except:
                            pass


            else:
                if outputQueue.empty():
                    # Bored? Add a random param
                    #inputQueue.put( ( False, getRandomInputParams(config) ) )
                    outputQueue.put( getRandomOutputParams(config) )


    print("Saving before exit...")
    neuralNet.save( modelFile )
    saveCsv( csvName, trainingInputs, trainingOutputs)
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