import cv2
import numpy as np

image = cv2.imread(r"noise.jpg", 0)

# rows of kernel
r = 3
# columns of kernel
c = 3

kernel = np.ones((r, c))
kernel[:, :] = 255


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

def compareImages(image1, image2):
    newImage = np.zeros(image1.shape)
    row = len(image1)
    col = len(image1[0])
    for i in range(row):
        for j in range(col):
            if image1[i][j] != image2[i][j]:
                newImage[i][j] = 255

    return newImage


# first closing then opening
image1 = dilation(erosion(erosion(dilation(image, kernel), kernel), kernel), kernel)
# first opening then closing
image2 = erosion(dilation(dilation(erosion(image, kernel), kernel), kernel), kernel)
# opening then closing in order to remove both kinds of noise

cv2.imwrite('res_noise1.jpg', image1)
cv2.imwrite('res_noise2.jpg', image2)


#task 1 b

image3 = compareImages(image1,image2)
cv2.imwrite('comparison.jpg',image3)

#task 1 c

bound1 = image1 - erosion(image1,kernel)
bound2 = image2 - erosion(image2,kernel)

cv2.imwrite('res_bound1.jpg', bound1)
cv2.imwrite('res_bound2.jpg', bound2)