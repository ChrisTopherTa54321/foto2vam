# Generate training data from existing faces

from Utils.Face.encoded import EncodedFace
from Utils.Face.normalize import FaceNormalizer
from PIL import Image
import multiprocessing
import argparse
import glob
import os
import queue
import fnmatch

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)
    inputPath = args.inputPath
    #outputPath = args.outputPath
    numThreads = args.numThreads
    recursive = args.recursive
    fileFilter = args.filter
    debugPose = args.debugPose

    poolWorkQueue = multiprocessing.Queue()
    doneEvent = multiprocessing.Event()
    if numThreads > 1:
        pool = []
        for idx in range(numThreads):
            proc = multiprocessing.Process(target=worker_process_func, args=(idx, poolWorkQueue, doneEvent, args) )
            proc.start()
            pool.append( proc )
    else:
        pool = None
        doneEvent.set()


    # Read in all of the files from inputpath
    for root, subdirs, files in os.walk(inputPath):
        print("Entering directory {}".format(root))
        for file in fnmatch.filter(files, fileFilter):
            fileName = "{}.encoding".format( os.path.splitext(file)[0] )
            inputFile = os.path.join(root, file )
            outputFile = os.path.join( root, fileName )

            try:
                # If this doesn't throw an exception, then we've already made this encoding
                EncodedFace.createFromFile(outputFile)
            except:
                poolWorkQueue.put( (inputFile, outputFile ))
                if pool is None:
                    worker_process_func(0, poolWorkQueue, doneEvent, args)

        if not recursive:
            break

    print("Generator done!")
    doneEvent.set()
    if pool:
        for proc in pool:
            proc.join()



###############################
# Worker function for helper processes
###############################
def worker_process_func(procId, workQueue, doneEvent, args):
    print("Worker {} started".format(procId))
    if args.normalize:
        normalizer = FaceNormalizer(256)
    else:
        normalizer = None

    while not ( doneEvent.is_set() and workQueue.empty() ):
        try:
            work = workQueue.get(block=True, timeout=1)
            inputFile = work[0]
            outputFile = work[1]
            print("Worker thread {} to generate {}->{}".format(procId, inputFile,outputFile))
            try:
                if os.path.splitext(inputFile)[0].endswith("normalized"):
                    print( "Skipping already normalized image" )
                    continue
                image = Image.open(inputFile)
                if normalizer:
                    image = normalizer.normalize(image)
                    fileName = "{}_normalized.png".format( os.path.splitext(inputFile)[0])
                    image.save( fileName)

                encodedFace = EncodedFace(image, debugPose = args.debugPose )
                if encodedFace.getAngle() < 0:
                    print( "Mirroring image to face left")
                    encodedFace = EncodedFace( image.transpose(Image.FLIP_LEFT_RIGHT), debugPose = args.debugPose )
                encodedFace.saveEncodings(outputFile)
            except Exception as e:
                print("Failed to generate {} : {}".format(outputFile, str(e)))
        except queue.Empty:
            pass
    print("Worker {} done!".format(procId))

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing images files to encode", required=True)
    parser.add_argument('--filter', help="File filter to process. Defaults to *.png", default="*.png")
    #parser.add_argument('--outputPath', help="Directory to write output data to", default="output")
    parser.add_argument("--debugPose", action='store_true', default=False, help="Display landmarks and pose on each image")
    parser.add_argument("--recursive", action='store_true', default=False, help="Recursively enter directories")
    parser.add_argument("--normalize", action='store_true', default=False, help="Perform image normalization")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")
    parser.add_argument("--numThreads", type=int, default=1, help="Number of processes to use")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )