import cv2
import numpy as np
import matplotlib.pylab as plt

def checkDilation(extractedImage, kernel):
    for k in range(len(kernel)):
        for l in range(len(kernel[0])):
            if extractedImage[k, l] == kernel[k, l]:
                return True

    return False


def checkErosion(extractedImage, kernel):
    f = 0
    for k in range(len(kernel)):
        for l in range(len(kernel[0])):
            if extractedImage[k, l] == kernel[k, l]:
                f = f + 1

    # if everything is 255, f is equal to 9
    if f == 9:
        return True
    else:
        return False


def dilation(image, kernel):
    imageNew = np.zeros(image.shape)
    xLen = int(len(kernel) / 2)
    yLen = int(len(kernel[0]) / 2)
    for i in range(xLen, len(image) - xLen):
        for j in range(yLen, len(image[0]) - yLen):
            extractedImage = image[i - xLen: i + xLen + 1, j - yLen: j + yLen + 1]
            if checkDilation(extractedImage, kernel):
                imageNew[i, j] = 255
    return imageNew


def erosion(image, kernel):
    imageNew = np.zeros(image.shape)
    xLen = int(len(kernel) / 2)
    yLen = int(len(kernel[0]) / 2)
    for i in range(xLen, len(image) - xLen):
        for j in range(yLen, len(image[0]) - yLen):
            extractedImage = image[i - xLen: i + xLen + 1, j - yLen: j + yLen + 1]
            if checkErosion(extractedImage, kernel):
                imageNew[i, j] = 255

    return imageNew


# plot histogram
def plotHistogram(imageArray):
    intensities = {}
    for i in range(rows):
        for j in range(colums):
            if imageArray[i][j] > 0:
                intensity = imageArray[i][j]
                flag = False
                if intensity in intensities:
                    flag = True
                if flag:
                    intensities[intensity] = intensities[intensity] + 1
                else:
                    intensities[intensity] = 1

    return intensities


def drawRectangle(new_img, box_list):
    for coordinates in box_list:
        print("coordinates:",coordinates[0], "(" , coordinates[0][0],  ",", coordinates[1][1], ")", "(" ,
              coordinates[1][0],  ",", coordinates[0][1], ")", coordinates[1])
        cv2.rectangle(new_img, coordinates[0][::-1], coordinates[1][::-1], (0, 0, 255), 2)


# read the image
imageArray = cv2.imread(r'segment.jpg', 0)
rows = imageArray.shape[0]
colums = imageArray.shape[1]

hist = plotHistogram(imageArray)
# sort according to keys of the hist that is all intensities
containers = sorted(hist.items())
a, b = zip(*containers)
plt.plot(a, b)
plt.show()

# setting threshold
threshold = 197

imageSegmented = np.zeros((rows, colums))

for i in range(rows):
    for j in range(colums):
        if imageArray[i][j] >= threshold:
            imageSegmented[i][j] = 255

cv2.imwrite("task2bSegmentedImage.jpg",imageSegmented)

# rows of kernel
r = 3
# columns of kernel
c = 3

kernel = np.ones((r, c))
kernel[:, :] = 255

imgNew = erosion(dilation(imageSegmented,kernel),kernel)

x = imgNew.shape[0]
y = imgNew.shape[1]
imgNew = imgNew / 255

img1 = np.zeros(shape=imgNew.shape)
img2 = np.zeros(shape=imgNew.shape)

sum1 = np.sum(imgNew, axis=0)
for i in range(len(sum1)):
    if sum1[i] < 2:
        img1[:, i] = 50

sum2 = np.sum(imgNew, axis=1)
for i in range(len(sum2)):
    if sum2[i] < 3:
        img1[i, :] = 50

for i in range(x):
    j = 0
    while j < y:
        if img1[i][j] != 50:
            l = 0
            while img1[i][j + l] == 0:
                l += 1
            if sum(imgNew[i][j:j + l]) == 0:
                img2[i, j:j + l] = 50
            j += l
        else:
            j += 1

for i in range(x):
    for j in range(y):
        img1[i][j] = max(img1[i][j], img2[i][j])


box = []
for x in range(img1.shape[0]):
    y = 0
    while y < img1.shape[1]:
        if img1[x][y] != 0:
            y += 1
        else:
            l = 0
            h = 0
            if img1[x + 10][y + l] == 0:
                while img1[x + 10][y + l] == 0:
                    l += 1
            else:
                while img1[x][y + l] == 0:
                    l += 1
            while img1[x + h][y] == 0:
                h += 1
            if l > 20 and h > 20:
                flag = True
                max = 10
                dummy = [(x, y), (x + h, y + l)]
                for line in box:
                    if line[0][0] - max <= dummy[0][0] <= line[1][0] + max and line[0][0] - max <= dummy[1][0] <= line[1][
                        0] + max \
                            and line[0][1] - max <= dummy[0][1] <= line[1][1] + max:
                        flag = False
                        break
                if flag:
                    box.append(dummy)
            y += l

drawRectangle(imageArray, box)
cv2.imwrite("task2bFinal.jpg", imageArray)