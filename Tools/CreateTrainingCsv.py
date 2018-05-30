# Generate training data from existing faces
from Utils.Training.config import Config
import multiprocessing
import queue
import argparse
import glob
import os
import csv
import fnmatch
import time

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputPath
    outputName = args.outputName
    numThreads = args.numThreads
    config = Config.createFromFile( args.configFile )


    poolWorkQueue = multiprocessing.Queue(maxsize=200)
    doneEvent = multiprocessing.Event()
    if numThreads > 1:
        pool = []
        for idx in range(numThreads):
            proc = multiprocessing.Process(target=worker_process_func, args=(idx, poolWorkQueue, doneEvent, config, args) )
            proc.start()
            pool.append( proc )
    else:
        pool = None
        doneEvent.set()


    # Read in all of the files from inputpath
    for root, subdirs, files in os.walk(inputPath):
        print("Generator entering directory {}".format(root))
        outCsvFile = os.path.join(root,outputName)
        poolWorkQueue.put( (root, outCsvFile) )
        if pool is None:
            worker_process_func(0, poolWorkQueue, doneEvent, config, args)

        if not args.recursive:
            break

    print("Generator done!")
    doneEvent.set()
    if pool:
        for proc in pool:
            proc.join()



###############################
# Worker function for helper processes
###############################
def worker_process_func(procId, workQueue, doneEvent, config, args):
    print("Worker {} started".format(procId))
    while not ( doneEvent.is_set() and workQueue.empty() ):
        try:
            work = workQueue.get(block=True, timeout=1)
            dirPath = work[0]
            outCsvFile = work[1]
            outFile = None
            numCreated = 0
            start = time.time()
            try:
                globPath = os.path.join( dirPath, "*.json")
                for file in glob.glob( globPath ):
                    try:
                        basename = os.path.splitext(file)[0]
                        # Check if we have all support encodings for this json
                        relatedFilesGlob = "{}*".format(basename)
                        relatedFiles = []
                        for rfile in glob.glob( relatedFilesGlob ):
                            relatedFiles.append( rfile )

                        # Have all files? Convert them to CSV
                        outRow = config.generateParams( relatedFiles )
                        if outFile is None:
                            print( "Worker {} creating {}".format(procId, outCsvFile))
                            outFile = open( outCsvFile, 'w' )
                            writer = csv.writer( outFile, lineterminator='\n')
                            shape = config.getShape()
                            print( "#{},{},{}".format( args.configFile, shape[0], shape[1] ), file=outFile )
                        writer.writerow( outRow )
                        numCreated += 1

                    except Exception as e:
                        pass
                        #print( "Failed to generate CSV from {} - {}".format( file, str(e)))
                print( "Worker {} done with {} ({} entries took {} seconds, at {} entries/second)".format(procId, outCsvFile, numCreated, time.time() - start, numCreated/( time.time() - start ) ) )

            except Exception as e:
                print("Worker {} failed generating {} : {}".format(procId, outCsvFile, str(e)))

        except queue.Empty:
            pass
    print("Worker {} done!".format(procId))


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing JSON and encoding files", required=True)
    parser.add_argument('--configFile', help="File with training data generation parameters", required=True)
    parser.add_argument("--recursive", action='store_true', default=False, help="Iterate to subdirectories of input path")
    parser.add_argument("--outputName", help="Name of CSV file to create in each directory")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")
    parser.add_argument("--numThreads", type=int, default=1, help="Number of processes to use")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )