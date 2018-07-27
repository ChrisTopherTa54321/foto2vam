import argparse
import os
import glob
from Utils.Vam.window import VamWindow
import time
from win32api import GetKeyState
from win32con import VK_CAPITAL, VK_SCROLL
from Utils.Face.vam import VamFace
import multiprocessing
import queue
import fnmatch
from collections import deque

# Set DPI Awareness  (Windows 10 and 8). Makes GetWindowRect return pxiel coordinates
import ctypes
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)


###############################
# Run the program
#
def main( args ):
    global testJsonPath
    global outputPath
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputJsonPath
    recursive = args.recursive
    fileFilter = args.filter

    print( "Input path: {}\n\n".format( inputPath ) )

    # Initialize the Vam window
    vamWindow = VamWindow( pipe = "foto2vamPipe" )

    angles = [0, 35, 70]
    skipCnt = 0
    screenshots = deque(maxlen=2)
    for root, subdirs, files in os.walk(inputPath):
        print("Entering directory {}".format(root))
        for file in fnmatch.filter(files, fileFilter):
            try:
                anglesToProcess = [] + angles
                for angle in angles:
                    fileName = "{}_{}.png".format( file, angle)
                    fileName = os.path.join( root,fileName)
                    if os.path.exists(fileName) or os.path.exists("{}.failed".format(fileName) ):
                        anglesToProcess.remove(angle)

                if len(anglesToProcess) == 0:
                    skipCnt += 1
                    #print("Nothing to do for {}".format(file))
                    continue
                print("Processing {} (after skipping {})".format(file, skipCnt))
                skipCnt = 0

                if (GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL)):
                    print("WARNING: Suspending script due to Caps Lock or Scroll Lock being on. Push CTRL+PAUSE/BREAK or mash CTRL+C to exit script.")
                    while GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL):
                        time.sleep(1)
                # Get screenshots of face and submit them to worker threads
                inputFile = os.path.join( os.path.abspath(root), file )
                vamWindow.loadLook(inputFile, anglesToProcess )
                continue
            except Exception as e:
                print("Failed to process {} - {}".format(file, str(e)))

        if not recursive:
            break

    print("Generator done!")


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate images for json" )
    parser.add_argument('--inputJsonPath', help="Directory containing json files to start with", required=True)
    parser.add_argument('--filter', help="File filter to process. Defaults to *.json", default="*.json")
    parser.add_argument('--normalizeSize', type=int, help="Size of normalized output. Defaults to 500", default=500)
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")
    parser.add_argument("--recursive", action='store_true', default=False, help="Recursively enter directories")
    return parser.parse_args()

###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )