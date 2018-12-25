UBIT = 'shivamag'
import cv2
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

def printLine(img1,img2,lines,pts1,pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# reading images
imageLeft = cv2.imread(r'mountain1.jpg')
imageRight = cv2.imread(r'mountain2.jpg')

sift = cv2.xfeatures2d.SIFT_create()
# detect key points in images
keyPointL, descriptorL = sift.detectAndCompute(imageLeft, None)
keyPointR, descriptorR = sift.detectAndCompute(imageRight, None)

# draw key points on the images
keyPointLeft = cv2.drawKeypoints(imageLeft, keyPointL, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keyPointRight = cv2.drawKeypoints(imageRight, keyPointR, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# write back images
cv2.imwrite('task1_sift1.jpg', keyPointLeft)
cv2.imwrite('task1_sift2.jpg', keyPointRight)

# finding matches
matcherObject = cv2.BFMatcher()
matches = matcherObject.knnMatch(descriptorL, descriptorR, k=2)

# finding matches with in distance 0.75
finalMatches = []
pointsL = []
pointsR = []
for a, b in matches:
    if a.distance < 0.75*b.distance:
        finalMatches.append(a)
        pointsL.append(keyPointL[a.queryIdx].pt)
        pointsR.append(keyPointR[a.trainIdx].pt)

# draw matches on image
mountainKnn = cv2.drawMatches(imageLeft,keyPointL,imageRight,keyPointR,finalMatches,outImg=None, matchColor=None,
                              singlePointColor=None, matchesMask=None,flags=2)

cv2.imwrite('task1_matches_knn.jpg', mountainKnn)

pointsL = np.int32(pointsL)
pointsR = np.int32(pointsR)

homographyMatrix, mask = cv2.findHomography(pointsL, pointsR, method=cv2.RANSAC)

print(homographyMatrix)

#print(inliers)
inliers = mask.ravel().tolist()

indexes = np.random.randint(0, len(inliers), 10)
inlierNew = []
matchesNew = []
for i in indexes:
    inlierNew.append(inliers[i])
    matchesNew.append(finalMatches[i])

# draw matches on image only inliers
mountainInliers = cv2.drawMatches(imageLeft,keyPointL,imageRight,keyPointR,matchesNew ,outImg=None, matchColor=(255,0,0),
                                  singlePointColor=None, matchesMask=inlierNew,flags=2)

cv2.imwrite('task1_matches.jpg', mountainInliers)

row1 = imageLeft.shape[0]
col1 = imageLeft.shape[1]
row2 = imageRight.shape[0]
col2 = imageRight.shape[1]

# reshape array using the corners of the images
reshape1 = np.float32([[0,0], [0,row1], [col1, row1], [col1,0]]).reshape(-1, 1, 2)
reshape2 = np.float32([[0,0], [0,row2], [col2, row2], [col2,0]]).reshape(-1, 1, 2)

# calculate transform on the right image using homography matrix
reshape2 = cv2.perspectiveTransform(reshape2, homographyMatrix)
# join both the arrays the arrays
concatenatedMatrix = np.concatenate((reshape1, reshape2), axis=0)

# calculating minimum and maximum
[xMin, yMin] = np.int32(concatenatedMatrix.min(axis=0).ravel() - 0.5)
[xMax, yMax] = np.int32(concatenatedMatrix.max(axis=0).ravel() + 0.5)

translationDistance = [-xMin, -yMin]
homographyTransnslation = np.array([[1, 0, translationDistance[0]], [0, 1, translationDistance[1]], [0, 0, 1]])

dotProduct = homographyTransnslation.dot(homographyMatrix)

# warp the image1
resultImage = cv2.warpPerspective(imageLeft,dotProduct , (xMax - xMin, yMax - yMin))

# concatenate with right image
resultImage[translationDistance[1]:row1+translationDistance[1],translationDistance[0]:col1+translationDistance[0]] = imageRight

cv2.imwrite('task1_pano.jpg', resultImage)