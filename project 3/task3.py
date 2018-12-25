import cv2
import numpy as np
import math

# perform erosion
def erosion(image, kernel):
    imageNew = np.zeros(image.shape)
    xLen = int(len(kernel) / 2)
    yLen = int(len(kernel[0]) / 2)
    for i in range(xLen, len(image) - xLen):
        for j in range(yLen, len(image[0]) - yLen):
            extractedImage = image[i - xLen: i + xLen + 1, j - yLen: j + yLen + 1]
            if checkErosion(extractedImage, kernel):
                imageNew[i][j] = 255

    return imageNew


# check erosion
def checkErosion(extractedImage, kernel):
    f = 0
    for k in range(len(kernel)):
        for l in range(len(kernel[0])):
            if extractedImage[k][l] >= kernel[k][l]:
                f = f + 1

    # if everything is 255, f is equal to 9
    if f == 9:
        return True
    else:
        return False


# compute hough space
def houghSpace(edgesImg,rohValues,thetaValues):
    newImage = np.zeros((rohValues, thetaValues))
    for i in range(rows):
        for j in range(colums):
            if edgesImg[i][j] == 255:
                #print(i,j)
                for theta in range(thetaValues):
                    sinValue = math.sin(math.radians(theta))
                    cosValue = math.cos(math.radians(theta))
                    roh = int(i * cosValue + j * sinValue)
                    if roh < 0:
                        roh = roh * (-1)
                    newImage[roh][theta] += 1

    return newImage


# convert to list
def arrayToList(array,x,y):
    list1 = []
    for i in range(x):
        for j in range(y):
            list1.append(array[i][j])
            #print(array[i][j])

    return list1


# find roh and thetha
def findRT(img,list1,x,y):
    l = len(list1)
    newDic = {}
    for i in range(x):
        for j in range(y):
            for k in range(l):
                if list1[k] == img[i][j]:
                    newDic[i] = j

    return newDic



# read the image
originalImage = cv2.imread('hough.jpg')
imageArray = cv2.imread(r'hough.jpg', 0)
rows = imageArray.shape[0]
colums = imageArray.shape[1]

# form kernel for vertical line
vertKernel = [[0,1,0],[0,1,0],[0,1,0]]
diaKernel = [[1,0,0],[0,1,0],[0,0,1]]

# use canny edge detection system
edges = cv2.Canny(imageArray, 50, 150, apertureSize= 3)

verticalEdgesImage = erosion(edges,vertKernel)
verticalEdgesImage = erosion(verticalEdgesImage,vertKernel)
verticalEdgesImage = erosion(verticalEdgesImage,vertKernel)
#cv2.imwrite('verticalEdgesImage.jpg',verticalEdgesImage)

rohValues = int((2 * (((rows ** 2) + (colums ** 2)) ** 0.5)) + 1)
thetaValues = 180

# calculate hough value
houghSpaceMatrix = houghSpace(verticalEdgesImage,rohValues, thetaValues)
listNew = arrayToList(houghSpaceMatrix,rohValues,thetaValues)

listNew.sort()

length = len(listNew)

thresholdList = []

# storing values in a new list
for i in range(11):
    thresholdList.append(listNew[length-i-1])

rohThetaValues = findRT(houghSpaceMatrix,thresholdList,rohValues,thetaValues)

redLines = originalImage

for x in range(rows):
    for y in range(colums):
        for r,t in rohThetaValues.items():
            value = int(((-1)/math.tan(math.radians(t)))*x + r/math.sin(math.radians(t)))
            if y == value:
                redLines[x][y] = [0,0,255]

cv2.imwrite('red_line.jpg',redLines)


#task 2b finding diagonals
diagonalsEdgesImage = erosion(edges,diaKernel)
cv2.imwrite('diagonalsEdgesImage.jpg',diagonalsEdgesImage)

# calculate hough value
houghSpaceMatrix1 = houghSpace(diagonalsEdgesImage,rohValues, thetaValues)
listNew1 = arrayToList(houghSpaceMatrix1,rohValues,thetaValues)

listNew1.sort()

length = len(listNew1)

thresholdList1 = []

# finding maximum value
for m in range(35):
    thresholdList1.append(listNew1[length-m-1])


#print(thresholdList1)
rohThetaValues1 = findRT(houghSpaceMatrix1,thresholdList1,rohValues,thetaValues)
#print(rohThetaValues1)

blueLines = originalImage

for x in range(rows):
    for y in range(colums):
        for r,t in rohThetaValues1.items():
            #print('inside for')
            value = int(((-1)/math.tan(math.radians(t)))*x + r/math.sin(math.radians(t)))
            #print(value)
            #print(y)
            if y == value:
                if(edges[x][y] >= 1):
                    blueLines[x][y] = [0,0,255]

cv2.imwrite('blue_lines.jpg',blueLines)