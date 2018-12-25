import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculateDistance(x, y, clusX, clusY):
    distance = (((clusX-x)**2) + ((clusY - y)**2))**0.5
    return distance


def plotTriangles(x, y, color):
    plt.scatter(x, y, s=50, marker='^', facecolors='none', edgecolors=color)
    plt.text(x+0.025, y+0.025, '%s, %s' % (str(x)[0:4], str(y)[0:4]), fontsize=6)


def plotCircles(x, y, color):
    plt.scatter(x, y, s=50, marker='o', c=color)
    plt.text(x+0.025, y+0.025, '%s, %s' % (str(x)[0:4], str(y)[0:4]), fontsize=6)


def computeNewCenter(clusterList):
    x = 0
    y = 0
    l = len(clusterList)
    for i in range(l):
        x = x + clusterList[i][0]
        y = y + clusterList[i][1]

    return x/l, y/l


xMatrix = [[5.9,3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8],
           [6.0, 3.0]]
clusterCenter1 = [6.2, 3.2] # red
clusterCenter2 = [6.6, 3.7] # green
clusterCenter3 = [6.5, 3.0] # blue

cluster1 = []
cluster2 = []
cluster3 = []
classificationVector = []

for i in range(len(xMatrix)):
    d1 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter1[0], clusterCenter1[1])
    d2 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter2[0], clusterCenter2[1])
    d3 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter3[0], clusterCenter3[1])

    if(d1<d2 and d1<d3):
        cluster1.append(xMatrix[i])
        classificationVector.append(0)
    elif(d2<d1 and d2<d3):
        cluster2.append(xMatrix[i])
        classificationVector.append(1)
    else:
        cluster3.append(xMatrix[i])
        classificationVector.append(2)


for i in range(len(cluster1)):
    plotTriangles(cluster1[i][0], cluster1[i][1], "red")

for i in range(len(cluster2)):
    plotTriangles(cluster2[i][0], cluster2[i][1], "green")

for i in range(len(cluster3)):
    plotTriangles(cluster3[i][0], cluster3[i][1], "blue")

# display classification vector
print('classificationVector 3.1')
print(classificationVector)

plt.scatter(clusterCenter1[0], clusterCenter1[1], s=50, marker='o', c='r')
plt.text(clusterCenter1[0]+0.025, clusterCenter1[1]+0.025, '%s, %s' % (str(clusterCenter1[0])[0:4],
                                                                       str(clusterCenter1[1])[0:4]), fontsize=6)

plt.scatter(clusterCenter2[0], clusterCenter2[1], s=50, marker='o', c='g')
plt.text(clusterCenter2[0]+0.025, clusterCenter2[1]+0.025, '%s, %s' % (str(clusterCenter2[0])[0:4],
                                                                       str(clusterCenter2[1])[0:4]), fontsize=6)

plt.scatter(clusterCenter3[0], clusterCenter3[1], s=50, marker='o', c='b')
plt.text(clusterCenter3[0]+0.025, clusterCenter3[1]+0.025, '%s, %s' % (str(clusterCenter3[0])[0:4],
                                                                       str(clusterCenter3[1])[0:4]), fontsize=6)


plt.savefig('task3_iter1_a.jpg')

plt.clf()

# start of 3.2
x1, y1 = computeNewCenter(cluster1)

x2, y2 = computeNewCenter(cluster2)

x3, y3 = computeNewCenter(cluster3)

for i in range(len(cluster1)):
    plotTriangles(cluster1[i][0], cluster1[i][1], "red")

for i in range(len(cluster2)):
    plotTriangles(cluster2[i][0], cluster2[i][1], "green")

for i in range(len(cluster3)):
    plotTriangles(cluster3[i][0], cluster3[i][1], "blue")

plotCircles(x1, y1, 'red')
plotCircles(x2, y2, 'green')
plotCircles(x3, y3, 'blue')

plt.savefig('task3_iter1_b.jpg')

plt.clf()

# start of 3.3
classificationVector = []
clusterCenter1New = [x1, y1]
clusterCenter2New = [x2, y2]
clusterCenter3New = [x3, y3]

cluster1New = []
cluster2New = []
cluster3New = []

for i in range(len(xMatrix)):
    d1 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter1New[0], clusterCenter1New[1])
    d2 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter2New[0], clusterCenter2New[1])
    d3 = calculateDistance(xMatrix[i][0], xMatrix[i][1], clusterCenter3New[0], clusterCenter3New[1])

    if(d1<d2 and d1<d3):
        cluster1New.append(xMatrix[i])
        classificationVector.append(0)
    elif(d2<d1 and d2<d3):
        cluster2New.append(xMatrix[i])
        classificationVector.append(1)
    else:
        cluster3New.append(xMatrix[i])
        classificationVector.append(2)


for i in range(len(cluster1New)):
    plotTriangles(cluster1New[i][0], cluster1New[i][1], "red")

for i in range(len(cluster2New)):
    plotTriangles(cluster2New[i][0], cluster2New[i][1], "green")

for i in range(len(cluster3New)):
    plotTriangles(cluster3New[i][0], cluster3New[i][1], "blue")

plt.scatter(clusterCenter1New[0], clusterCenter1New[1], s=50, marker='o', c='r')
plt.text(clusterCenter1New[0]+0.025, clusterCenter1New[1]+0.025, '%s, %s' % (str(clusterCenter1New[0])[0:4],
                                                                       str(clusterCenter1New[1])[0:4]), fontsize=6)

plt.scatter(clusterCenter2New[0], clusterCenter2New[1], s=50, marker='o', c='g')
plt.text(clusterCenter2New[0]+0.025, clusterCenter2New[1]+0.025, '%s, %s' % (str(clusterCenter2New[0])[0:4],
                                                                       str(clusterCenter2New[1])[0:4]), fontsize=6)

plt.scatter(clusterCenter3New[0], clusterCenter3New[1], s=50, marker='o', c='b')
plt.text(clusterCenter3New[0]+0.025, clusterCenter3New[1]+0.025, '%s, %s' % (str(clusterCenter3New[0])[0:4],
                                                                       str(clusterCenter3New[1])[0:4]), fontsize=6)


plt.savefig('task3_iter2_a.jpg')

plt.clf()

for i in range(len(cluster1New)):
    plotTriangles(cluster1New[i][0], cluster1New[i][1], "red")

for i in range(len(cluster2New)):
    plotTriangles(cluster2New[i][0], cluster2New[i][1], "green")

for i in range(len(cluster3New)):
    plotTriangles(cluster3New[i][0], cluster3New[i][1], "blue")

x1New, y1New = computeNewCenter(cluster1New)

x2New, y2New = computeNewCenter(cluster2New)

x3New, y3New = computeNewCenter(cluster3New)

plotCircles(x1New, y1New, 'red')
plotCircles(x2New, y2New, 'green')
plotCircles(x3New, y3New, 'blue')

print('classificationVector 3.3')
print(classificationVector)

classificationVector = []
classificationVector.append([x1New,y1New])
classificationVector.append([x2New,y2New])
classificationVector.append([x3New,y3New])
print('Updated Mu value 3.3')
print(classificationVector)

plt.savefig('task3_iter2_b.jpg')