import argparse
import os
import glob
from Utils.Vam.window import VamWindow
from Utils.Face.normalize import FaceNormalizer
from PIL import Image
import time
from win32api import GetKeyState
from win32con import VK_CAPITAL, VK_SCROLL
from Utils.Face.encoded import EncodedFace
from Utils.Face.vam import VamFace

# Set DPI Awareness  (Windows 10 and 8). Makes GetWindowRect return pxiel coordinates
import ctypes
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)

#temp
testJsonPath = None
vamWindow = VamWindow()
vamWindow.setClickLocations()

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
    outputPath = args.outputPath

    testJsonPath = args.testJsonPath


    print( "Input path: {}\nOutput path: {}\n\n".format( inputPath, outputPath ) )

    # Initialize the Vam window
    window = VamWindow()

    # Locating the buttons via image comparison does not reliably work. These coordinates are the 'Window' coordinates
    # found via AutoHotKey's Window Spy, cooresponding to the Load Preset button and the location of the test file
    window.setClickLocations([(130,39), (248,178)])

    evalNum = 0
    angles = [0, 35, 65]
    face_images = {}
    for entry in glob.glob(os.path.join(inputPath, '*.json')):
        print("Processing {}".format(entry))
        try:
            alreadyDone = False
            for angle in angles:
                fileName = "{}_angle{}.png".format( os.path.splitext(os.path.basename(entry))[0], angle)
                fileName = os.path.join( outputPath,fileName) 
                if os.path.exists(fileName):
                    alreadyDone = True
                    print("Output file already exists. Skipping.")
                    break

            if not alreadyDone:
                inputFace = VamFace(entry)
                face_images = evaluate_get_face_images(inputFace, angles)
                for angle,face in face_images.items():
                    fileName = "{}_angle{}.png".format( os.path.splitext(os.path.basename(entry))[0], angle)
                    face.save( os.path.join( outputPath, fileName ))
        except:
            print("Failed to process {}".format(entry))
        evalNum += 1



###############################
# Return images of the test faces
###############################
def evaluate_get_face_images(face, angles):
    global testJsonPath
    global vamWindow
    # Give the user a chance to interrupt the process before we hijack the mouse
    if (GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL)):
        print("WARNING: Suspending script due to Caps Lock or Scroll Lock being on. Push CTRL+PAUSE/BREAK or mash CTRL+C to exit script.")
        while GetKeyState(VK_CAPITAL) or GetKeyState(VK_SCROLL):
            time.sleep(1)

    face_images = {}
    for angle in angles:
        face.setRotation(angle)
        face.save( testJsonPath )
        vamWindow.loadLook()
        time.sleep(.3)
        img = vamWindow.getScreenShot()

        try:
            faceNormalizer = FaceNormalizer(256)
            normalized = faceNormalizer.normalize(img)
            face_images[angle] = normalized
        except:
            print( "Failed to locate face for angle {}".format(angle))

    return face_images


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate images for json" )
    parser.add_argument('--inputJsonPath', help="Directory containing json files to start with", required=True)
    parser.add_argument('--outputPath', help="Directory to write output data to", default="output")
    parser.add_argument('--testJsonPath', help="Directory where test JSON will be stored", default="test")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")
    return parser.parse_args()

###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )