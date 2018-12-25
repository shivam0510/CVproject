import cv2
import numpy as np

image = cv2.imread(r'C:/Users/Shivam Agarwal/Documents/homework stuff/cvip/task3/neg_3.jpg')
templateImage = cv2.imread(r'C:/Users/Shivam Agarwal/Documents/homework stuff/cvip/task3/template.png')

image1 = cv2.GaussianBlur(image,( 3, 3 ), 1)
templateImageBlur = cv2.GaussianBlur(templateImage,(3,3),1)
imageLaplacian = cv2.Laplacian(image1,cv2.CV_8U)
templateImageLaplacian = cv2.Laplacian(templateImageBlur, cv2.CV_8U)
finalTemp = cv2.GaussianBlur(templateImageLaplacian,(3,3),0)
extra, newImage = cv2.threshold(imageLaplacian, 24, 255.0, cv2.THRESH_BINARY)
newImage1 = cv2.GaussianBlur(newImage,(3,3),0)
extra, finalImage = cv2.threshold(newImage1, 30, 255.0, cv2.THRESH_BINARY)

scales = [0.4,0.5,0.6,0.7,0.8]
differentScalesTemp = []
result = []
#list of different scales of template
for i in scales:
	differentScalesTemp.append(cv2.resize(finalTemp, (0, 0), fx=i, fy=i))

#list of all the values of matching based on template
for index in differentScalesTemp:
	result.append(cv2.matchTemplate(finalImage,index, cv2.TM_CCORR_NORMED))

bestMaxVal=0
bestMaxLoc=(0,0)
index = 0
count = 0
#finding best value	
for val in result:
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(val)
	count = count+1
	if(maxVal > bestMaxVal):
		bestMaxVal = maxVal
		bestMaxLoc = maxLoc
		index = count
	

finalMatch = result[index]
width = differentScalesTemp[index].shape[0]
height = differentScalesTemp[index].shape[1]

#atleast this much match should be there
threshold = 0.6

location = np.where(finalMatch>= threshold)

for pt in zip(*location[::-1]):
    cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (255,0,0),2)

cv2.destroyAllWindows()
cv2.imshow('image',image)
cv2.waitKey(0)