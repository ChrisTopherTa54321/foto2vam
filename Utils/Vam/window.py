# Class to interact with VAM window

import win32gui
import win32con
import pyautogui
import win32pipe;
import win32file;
import pywintypes;
import os
import time
import json

# convert a rect of (left, top, right, bottom) to (x, y, w, h)
def _rect2xywh(rect):
    left,top,right,bottom = rect
    return (left, top, right-left, bottom-top )


class VamWindow:
    _wHndl = 0
    _pipe = None
    _idx = 0

    def __init__(self, pipe = None ):
        if pipe != None:
            try:
                self._pipe = win32file.CreateFile(
                        r'\\.\\pipe\\' + pipe,
                        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                        0,
                        None,
                        win32file.OPEN_EXISTING,
                        0,
                        None
                        )
            except pywintypes.error as e:
                print( str(e) );

        self._getVamHndl()

        self._clickLocations = None
        if 0 == self._getVamHndl():
            raise Exception("Failed to get handle to VaM window! Is VaM running?")


    def _getVamHndl(self):
        if not win32gui.IsWindow(self._wHndl):
            def callback(hwnd, hwnds):
                if win32gui.GetWindowText(hwnd) == "VaM":
                    hwnds.append(hwnd)

            hwnds = []
            win32gui.EnumWindows( callback, hwnds )
            self._wHndl = hwnds[self._idx]
        return self._wHndl


    def _getVamRect(self):
        return win32gui.GetWindowRect(self._getVamHndl())

    def setClickLocations(self, clickLocations = [(130,39), (248,178)]):
        self._clickLocations = clickLocations

    def focus(self):
        delay = None # if the window wasn't already active we want to give it time to focus
        hwnd = self._getVamHndl()

        # Is the window minimized?
        if win32gui.IsIconic( hwnd ):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            delay = "Window was minimized"

        # Delay if the window wasn't already active
        if hwnd != win32gui.GetForegroundWindow():
            delay = "Window was not in focus"

        win32gui.ShowWindow(hwnd, 5)
        win32gui.SetForegroundWindow( hwnd )


        if delay:
            print( "Delaying because: {}".format( delay ))
            time.sleep(.2)


    # Find an image in a region. Originalyl used for loadLook, but is very unreliable
    def locateInWindow(self, image, region=None, entireScreen=False):
        wX,wY,wW,wH = _rect2xywh(self._getVamRect() )

        # If a region was supplied then convert the window-relative coordinates to screen coordinates
        if not region is None:
            print( "Searching in region {}".format(region))
            rX,rY,rW,rH = region
            wX += rX
            wY += rY
            wW = min( rW, wW - rX )
            wH = min( rH, wH - rY )

        windowRegion = ( wX, wY, wW, wH )
        ret = pyautogui.locate( image, pyautogui.screenshot("haystack.png", region=windowRegion) )

        # Found a result in specified region. Convert to window coordinates
        if ret and region:
            retX,retY,retW,retH = ret
            ret = ( retX + rX, retY + rY, retW + rW, retH + rH )

        # If not found in region then search entire screen
        if ret is None and entireScreen:
            print( "Switching to entire screen")
            return self.locateInWindow( image )
        print("Returning: {}".format(ret))
        return ret


    def clickInWindow(self, x, y = None ):
        if y is None and type(x) == tuple:
            y = x[1]
            x = x[0]
        offsetX, offsetY, _, _ = self._getVamRect()
        pyautogui.click( x + offsetX, y + offsetY )


    # Click the buttons to load the look.
    # ClickLocations is an array of (x,y) tuples which will be clicked in order
    def clickLoadLook(self):
        self.focus()
        # If coordinates were passed in, just click the mouse on them one after the other
        if self._clickLocations:
            for coords in self._clickLocations:
                self.clickInWindow( coords )
            return


    def loadLook(self, jsonPath, angles = [0]):
        if self._pipe is not None:
            self.loadLookPipe( jsonPath, self._pipe, angles )
        else:
            raise Exception("Todo: Reimplement window clicking. Specify a pipe for control")

    def loadLookPipe(self, jsonPath, pipe, angles ):
        msg = {};
        msg["cmd"] = "screenshot";
        msg["angles"] = angles
        msg["json"] = jsonPath
        msg["outputPath"] = jsonPath
        msg["dimensions"] = [ 512, 512 ]
        win32file.WriteFile( pipe, (json.dumps(msg) + "<EOM>").encode() )

    def getScreenShot(self, region=None):
        self.focus()
        wX,wY,wW,wH = _rect2xywh(self._getVamRect() )

        # If a region was supplied then convert the window-relative coordinates to screen coordinates
        if not region is None:
            rX,rY,rW,rH = region
            wX += rX
            wY += rY
            wW = min( rW, wW - rX )
            wH = min( rH, wH - rY )
        else:
            wX,wY,wW,wH = _rect2xywh( self._getVamRect() )

        windowRegion = ( wX, wY, wW, wH )
        img = pyautogui.screenshot(region=windowRegion)
        return img
