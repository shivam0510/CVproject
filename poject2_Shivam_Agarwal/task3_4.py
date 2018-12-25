UBIT = 'shivamag'
import cv2
import random
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

def caldistance3points(x, y, z, clusR, clusG, clusB):
    distance = (((clusR-x)**2) + ((clusG - y)**2) + (clusB - z)**2)**0.5
    return distance

def calNewCluster(cluster):
    r = 0
    g = 0
    b = 0
    l = len(cluster)
    if l == 0:
        l = 1
    for i in cluster:
        r = r + i[0]
        g = g + i[1]
        b = b + i[2]

    return r/l, g/l, b/l

def updateFinalImage(ci,index):
    r = index[0]
    c = index[1]
    finalImage1[r, c, 0] = clusterCentersList[ci][0]
    finalImage1[r, c, 1] = clusterCentersList[ci][1]
    finalImage1[r, c, 2] = clusterCentersList[ci][2]

def updateClusterCenters(counter):
    for i in range(counter):
        r, g, b = calNewCluster(clusterLists[i])
        clusterCentersList[i]=[]
        clusterCentersList[i].append(r)
        clusterCentersList[i].append(g)
        clusterCentersList[i].append(b)


# start of 3.4
listOfN = [3, 5, 10, 20]

for n in listOfN:
    imageInput = cv2.imread(r'baboon.jpg')
    rows, cols, plane = imageInput.shape
    finalImage1 = imageInput
    clusterLists = []
    clusterLists = [[] for x in range(n)]
    clusterCentersList = []
    for i in range(n):
        clusterCentersList.append([np.random.randint(0, 255, 1)[0],np.random.randint(0, 255, 1)[0],np.random.randint(0, 255, 1)[0]])
    distances = []
    iterations = 20
    for a in range(iterations):

        # this will calculate distance for each pixel in image and move it in cluster
        for r in range(rows):
            for c in range(cols):
                distances = []
                for centre in clusterCentersList:
                    dist = caldistance3points(imageInput[r, c, 0], imageInput[r, c, 1],
                                              imageInput[r, c, 2],centre[0], centre[1], centre[2])
                    distances.append(dist)
                clusterLists[distances.index(min(distances))].append(imageInput[r,c])
                if(a == iterations-1):
                    updateFinalImage(distances.index(min(distances)),[r,c])
        updateClusterCenters(n)

    if(n==3):
        cv2.imwrite('task3_baboon_3.jpg', finalImage1)
    elif(n == 5):
        cv2.imwrite('task3_baboon_5.jpg', finalImage1)
    elif (n == 10):
        cv2.imwrite('task3_baboon_10.jpg', finalImage1)
    elif (n == 20):
        cv2.imwrite('task3_baboon_20.jpg', finalImage1)