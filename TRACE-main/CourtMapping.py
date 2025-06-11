# from numpy import float32, maximum, minimum, amax
from numpy import float32, zeros,  uint8, shape, array, squeeze
from cv2 import VideoCapture, getPerspectiveTransform, warpPerspective, line, circle, rectangle, perspectiveTransform
from TraceHeader import videoFile, checkPath
import cv2;
video = VideoCapture(videoFile)
checkPath(videoFile)
frameWidth = int(video.get(3))
frameHeight = int(video.get(4))

widthP = int(967)
heightP = int(1585)

width = int(967)
height = int(1585)

ratio = 6.1/13.4
courtHeight = int(height * 0.6)
courtWidth = int(courtHeight * ratio)
yOffset = int((height - courtHeight) / 2)
xOffset = int((width - courtWidth) / 2)

courtTL = [xOffset,yOffset]
courtTR = [courtWidth+xOffset,yOffset]
courtBL = [xOffset,courtHeight+yOffset]
courtBR = [courtWidth+xOffset,courtHeight+yOffset]

def courtMap(frame, top_left, top_right, bottom_left, bottom_right):
    pts1 = float32([[top_left, top_right, bottom_left, bottom_right]])
    pts2 = float32([courtTL, courtTR, courtBL, courtBR])
    M = getPerspectiveTransform(pts1,pts2)
    dst = warpPerspective(frame,M,(width,height))
    return dst, M

def showLines(frame):
    cv2.rectangle(frame, (xOffset,yOffset), (courtWidth+xOffset,courtHeight+yOffset), (255, 255, 255), 2)
    centerX=xOffset+courtWidth//2
    cv2.line(frame, (centerX, yOffset), (centerX, yOffset + courtHeight), (255, 255, 255), 2)
   
    shortServiceLineY1 = yOffset + int(courtHeight * 0.35472)  
    cv2.line(frame, (xOffset, shortServiceLineY1), (xOffset + courtWidth, shortServiceLineY1), (255, 255, 255), 2)
    shortServiceLineY2 = yOffset + int(courtHeight * 0.64528)  
    cv2.line(frame, (xOffset, shortServiceLineY2), (xOffset + courtWidth, shortServiceLineY2), (255, 255, 255), 2)

    singlesOffset = int(courtWidth * (.46/6.1)) 
    leftSinglesX = xOffset + singlesOffset
    rightSinglesX = xOffset + courtWidth - singlesOffset
    cv2.line(frame, (leftSinglesX, yOffset), (leftSinglesX, yOffset + courtHeight), (255, 255, 255), 2)
    cv2.line(frame, (rightSinglesX, yOffset), (rightSinglesX, yOffset + courtHeight), (255, 255, 255), 2)
    backServiceLineY_top = yOffset + int(courtHeight * 0.0567)  
    backServiceLineY_bottom = yOffset + int(courtHeight * 0.9424)  
    cv2.line(frame, (leftSinglesX, backServiceLineY_top), (rightSinglesX, backServiceLineY_top), (255, 255, 255), 2)
    cv2.line(frame, (leftSinglesX, backServiceLineY_bottom), (rightSinglesX, backServiceLineY_bottom), (255, 255, 255), 2)


    return frame

def showPoint(frame, M, point):
    points = float32([[point]])
    transformed = perspectiveTransform(points, M)[0][0]
    circle(frame, (int(transformed[0]), int(transformed[1])), radius=0, color=(0, 0, 255), thickness=25)
    return frame

def givePoint(M, point):
    points = float32([[point]])
    transformed = perspectiveTransform(points, M)[0][0]
    return (int(transformed[0]), int(transformed[1]))
