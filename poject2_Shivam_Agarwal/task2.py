UBIT = 'shivamag'
import cv2
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))


def printLine(img,lines,pts1,pts2):
    row, col = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count1 = 20
    count2 = 255
    count3 = 100
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple([count1, count2, count3])
        xInit, yInit = map(int, [0, -r[2] / r[1]])
        xNew, yNew = map(int, [col, -(r[2] + r[0] * col) / r[1]])
        img = cv2.line(img, (xInit, yInit), (xNew, yNew), color, 1)
        img = cv2.circle(img, tuple(pt1), 5, color, -1)
        count2 = count2 - 10
        count1 = count1 + 25
        count3 = count3 + 17
    return img


# reading images
imageLeft = cv2.imread(r'tsucuba_left.png')
imageRight = cv2.imread(r'tsucuba_right.png')

sift = cv2.xfeatures2d.SIFT_create()
# detect key points in images
keyPointL, descriptorL = sift.detectAndCompute(imageLeft, None)
keyPointR, descriptorR = sift.detectAndCompute(imageRight, None)

# draw key points on the images
keyPointLeft = cv2.drawKeypoints(imageLeft, keyPointL, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keyPointRight = cv2.drawKeypoints(imageRight, keyPointR, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# write back images
cv2.imwrite('task2_sift1.jpg', keyPointLeft)
cv2.imwrite('task2_sift2.jpg', keyPointRight)

# finding matches
matcherObject = cv2.BFMatcher()
matches = matcherObject.knnMatch(descriptorL, descriptorR, k=2)

# finding matches with in distance 0.75
finalMatches = []
pointsL = []
pointsR = []
for a, b in matches:
    if a.distance < 0.75*b.distance:
        finalMatches.append([a])
        pointsL.append(keyPointL[a.queryIdx].pt)
        pointsR.append(keyPointR[a.trainIdx].pt)

# draw matches on image
tsucubaKnn = cv2.drawMatchesKnn(imageLeft,keyPointL,imageRight,keyPointR,finalMatches,outImg=None, matchColor=None, singlePointColor=None, matchesMask=None,flags=2)

cv2.imwrite('task2_matches_knn.jpg', tsucubaKnn)

pointsL = np.int32(pointsL)
pointsR = np.int32(pointsR)

# calculating fundamental matrix
fundamental, mask = cv2.findFundamentalMat(pointsL, pointsR, cv2.RANSAC)

print(fundamental)

# end of part 2 of 2

# selecting inlier points
pointsL = pointsL[mask.ravel() == 1]
pointsR = pointsR[mask.ravel() == 1]

# selecting 10 inlier match pairs
indexes = np.random.randint(0, len(pointsL),10).tolist()
inlierL = []
inlierR = []
for i in indexes:
    inlierL.append(pointsL[i])
    inlierR.append(pointsR[i])

inlierL = np.int32(inlierL)
inlierR = np.int32(inlierR)

grayL= cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
grayR= cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)
# epilines for left image points, drawing on right image
linesR = cv2.computeCorrespondEpilines(inlierL.reshape(-1, 1, 2), 1, fundamental)
linesR = linesR.reshape(-1, 3)
img1 = printLine(grayR, linesR, inlierR, inlierL)


# epilines for left image points, drawing on right image
linesL = cv2.computeCorrespondEpilines(inlierR.reshape(-1, 1, 2), 2, fundamental)
linesL = linesL.reshape(-1, 3)
img3 = printLine(grayL, linesL, inlierL, inlierR)

cv2.imwrite('task2_epi_right.jpg', img1)
cv2.imwrite('task2_epi_left.jpg', img3)

# start of disparity
stereo = cv2.StereoBM_create(numDisparities=48, blockSize=31)
imageDisparity = stereo.compute(grayL, grayR)
cv2.imwrite('task2_disparity.jpg', imageDisparity)