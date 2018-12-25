import cv2
import numpy as np

# read the image
imageArray = cv2.imread(r'point.jpg', 0)
rows = imageArray.shape[0]
colums = imageArray.shape[1]
count = 0

# sobel matrix
operator = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

image = np.ndarray(shape=(rows, colums))

dummyMatrix = np.ndarray(shape=(3, 3))
for i in range(rows):
    for j in range(colums):
        sumMatrix = 0
        if i == 0 or j == 0 or i == rows - 1 or j == colums - 1:
            if i == 0 and j == 0:
                dummyMatrix = [[0, 0, 0],
                               [0, imageArray[i][j], imageArray[i][j + 1]],
                               [0, imageArray[i + 1][j], imageArray[i + 1][j + 1]]]
            elif i == rows - 1 and j == colums - 1:
                dummyMatrix = [[imageArray[i - 1][j - 1], imageArray[i - 1][j], 0],
                               [imageArray[i][j - 1], imageArray[i][j], 0],
                               [0, 0, 0]]

            elif i == 0 and j == colums - 1:
                dummyMatrix = [[0, 0, 0],
                               [imageArray[i][j - 1], imageArray[i][j], 0],
                               [imageArray[i + 1][j - 1], imageArray[i + 1][j], 0]]

            elif i == rows - 1 and j == 0:
                dummyMatrix = [[0, imageArray[i - 1][j], imageArray[i - 1][j + 1]],
                               [0, imageArray[i][j], imageArray[i][j + 1]],
                               [0, 0, 0]]

            elif i == 0 and j != 0 and j != colums - 1:
                dummyMatrix = [[0, 0, 0],
                               [imageArray[i][j - 1], imageArray[i][j], imageArray[i][j + 1]],
                               [imageArray[i + 1][j - 1], imageArray[i + 1][j], imageArray[i + 1][j + 1]]]

            elif j == 0 and i != 0 and i != rows - 1:
                dummyMatrix = [[0, imageArray[i - 1][j], imageArray[i - 1][j + 1]],
                               [0, imageArray[i][j], imageArray[i][j + 1]],
                               [0, imageArray[i + 1][j], imageArray[i + 1][j + 1]]]

            elif i == rows - 1 and j != colums - 1 and j != 0:
                dummyMatrix = [[imageArray[i - 1][j - 1], imageArray[i - 1][j], imageArray[i - 1][j + 1]],
                               [imageArray[i][j - 1], imageArray[i][j], imageArray[i][j + 1]],
                               [0, 0, 0]]
            elif i != rows - 1 and i != 0 and j == colums - 1:
                dummyMatrix = [[imageArray[i - 1][j - 1], imageArray[i - 1][j], 0],
                               [imageArray[i][j - 1], imageArray[i][j], 0],
                               [imageArray[i + 1][j - 1], imageArray[i + 1][j], 0]]
        else:
            dummyMatrix = [[imageArray[i - 1][j - 1], imageArray[i - 1][j], imageArray[i - 1][j + 1]],
                           [imageArray[i][j - 1], imageArray[i][j], imageArray[i][j + 1]],
                           [imageArray[i + 1][j - 1], imageArray[i + 1][j], imageArray[i + 1][j + 1]]]

        for k in range(3):
            for l in range(3):
                sumMatrix = sumMatrix + dummyMatrix[k][l] * operator[k][l]

        if sumMatrix < 0:
            sumMatrix = sumMatrix * (-1)

        if sumMatrix > 125 and i > 40 and i < 300 and j !=0 and j !=356:
            image[i, j] = 255
            print('coordinates i='+str(i)+' j='+str(j))
            count = count+1


cv2.imwrite('afterOperator.jpg', image)
