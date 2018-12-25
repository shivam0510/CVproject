import cv2
import numpy as np
import math

#returns an half sized image of the input
def imageResize(inputImage):
	newRow = math.ceil((inputImage.shape[0])/2)
	newColum = math.ceil((inputImage.shape[1])/2)
	newImage = np.ndarray(shape = (newRow,newColum))
	for i in range(0,inputImage.shape[0],2):
		#if(i%2 == 0):
		for j in range(0,inputImage.shape[1],2):
				#if(j%2 ==0):
			row = int(i/2)
			colum = int(j/2)
			newImage[row][colum] = inputImage[i][j]
	
	return newImage

#returns an half sized color image of the input
def imageResizeColor(inputImage):
	newRow = math.ceil((inputImage.shape[0])/2)
	newColum = math.ceil((inputImage.shape[1])/2)
	#adding color element as the third index
	newImage = np.ndarray(shape = (newRow,newColum,inputImage.shape[2]))
	for i in range(0,inputImage.shape[0],2):
		#if(i%2 == 0):
		for j in range(0,inputImage.shape[1],2):
				#if(j%2 ==0):
			row = int(i/2)
			colum = int(j/2)
			newImage[row][colum][0] = inputImage[i][j][0]
			newImage[row][colum][1] = inputImage[i][j][1]
			newImage[row][colum][2] = inputImage[i][j][2]
	
	return newImage

#returns gausian kernel for sigma received
def developGausian(sigma1):
	gausianKernel = np.ndarray(shape = (7,7))
	flipedGausianKernel = np.ndarray(shape = (7,7))
	x = -3
	y = 3
	for i in range(7):
		for j in range(7):
			x = x + j
			power = (-1 * ((x*x) + (y*y)))/(2*(sigma1*sigma1))
			divideBy = 2*(math.pi)*(sigma1*sigma1)
			numerator = math.exp(power)
			gausianKernel[i][j] = numerator/divideBy
		y = y-1
		x = -3
		
	for row in range(7):
		for col in range(7):
			flipedGausianKernel[row][col] = gausianKernel[6-row][6-col]
	
	return flipedGausianKernel

#return a image chunk
def imageChunk(image,startRow,startCol):
	dummyMatrix = np.ndarray(shape = (7,7))
	for i in range(startRow,startRow+7):
		for j in range(startCol,startCol+7):
			dummyMatrix[i-startRow][j-startCol] = image[i][j]
			
	return dummyMatrix
	
#return a image chunk
def imageChunkSmall(image,startRow,startCol):
	dummyMatrix = np.ndarray(shape = (3,3))
	for i in range(startRow,startRow+3):
		for j in range(startCol,startCol+3):
			dummyMatrix[i-startRow][j-startCol] = image[i][j]
			
	return dummyMatrix
	
#returns convolution values
def convolutionValue(dummyMatrix,flipedGausianKernel):
	sum = 0
	for i in range(7):
		for j in range(7):
			sum = sum + dummyMatrix[i][j]*flipedGausianKernel[i][j]
			
	return sum
	
#return convoluted matrix
def convulateImage(image,flipedGausianKernel):
	convolutedImage = np.ndarray(shape = (image.shape[0],image.shape[1]))
	for row in range(3,image.shape[0]-3):
		for col in range(3,image.shape[1]-3):
			dummyMatrix = imageChunk(image,row-3,col-3)
			convolutedImage[row][col] = convolutionValue(dummyMatrix,flipedGausianKernel)
	
	return convolutedImage
			
#returns normalised image
def normalisedImage(image):
	minH = 0
	maxH = 0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if (minH > image[i][j]):
				minH = image[i][j]
			if (maxH < image[i][j]):
				maxH = image[i][j]
	image1 = (image - minH) / (maxH - minH)
	return image1
			
	
#retuns min and max value of dogs
def calMinMax(mat1,mat2,mat3):
	arrayList = []
	minMax = mat2[1][1]
	for i in range(3):
		for j in range(3):
			arrayList.append(mat1[i][j])
			if(i!=1 and j!=1):
				arrayList.append(mat2[i][j])
			arrayList.append(mat3[i][j])
	
	minimum = min(arrayList)
	maximum = max(arrayList)
	
	if(minMax<minimum or maximum<minMax):
		return True
	else:
		return False
		
#prints 5 key points for image
def printPoints(image):
	k=0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if(image[i][j] != 0):
				print('row '+str(i)+' col '+str(j))
				if(k==5):
					break
				else:
					k = k+1
			
				
imageArray = cv2.imread(r'C:/Users/Shivam Agarwal/Documents/homework stuff/cvip/task2.jpg',0)
imageColor = cv2.imread('C:/Users/Shivam Agarwal/Documents/homework stuff/cvip/task2.jpg')

imageOctave1 = imageArray
result1_1 = np.array(imageColor)
result1_2 = np.array(imageColor)
gausianKernelFliped1 = developGausian(1 / math.sqrt(2))
gausianKernelFliped2 = developGausian(1)
gausianKernelFliped3 = developGausian(math.sqrt(2))
gausianKernelFliped4 = developGausian(2)
gausianKernelFliped5 = developGausian(2*math.sqrt(2))

imageConvoluted1_1 = convulateImage(imageOctave1,gausianKernelFliped1)
imageConvoluted1_2 = convulateImage(imageOctave1,gausianKernelFliped2)
imageConvoluted1_3 = convulateImage(imageOctave1,gausianKernelFliped3)
imageConvoluted1_4 = convulateImage(imageOctave1,gausianKernelFliped4)
imageConvoluted1_5 = convulateImage(imageOctave1,gausianKernelFliped5)

dog1_1 = imageConvoluted1_1 - imageConvoluted1_2
dog1_2 = imageConvoluted1_2 - imageConvoluted1_3
dog1_3 = imageConvoluted1_3 - imageConvoluted1_4
dog1_4 = imageConvoluted1_4 - imageConvoluted1_5

keyPointImage1_1 = np.ndarray(shape = (imageConvoluted1_1.shape[0],imageConvoluted1_1.shape[1]))
keyPointImage1_2 = np.ndarray(shape = (imageConvoluted1_1.shape[0],imageConvoluted1_1.shape[1]))

for i in range(3,imageConvoluted1_1.shape[0]-6):
	for j in range(3,imageConvoluted1_1.shape[1]-6):
		mat1 = imageChunkSmall(dog1_1,i,j)
		mat2 = imageChunkSmall(dog1_2,i,j)
		mat3 = imageChunkSmall(dog1_3,i,j)
		if(calMinMax(mat1,mat2,mat3)):
			keyPointImage1_1[i+1][j+1] = dog1_2[i+1][j+1]
		
		mat4 = imageChunkSmall(dog1_4,i,j)
		if(calMinMax(mat2,mat3,mat4)):
			keyPointImage1_2[i+1][j+1] = dog1_3[i+1][j+1]

for i in range(result1_1.shape[0]):
	for j in range(result1_1.shape[1]):
		if(keyPointImage1_1[i][j]!=0):
			result1_1[i][j][:] = 255
		if(keyPointImage1_2[i][j]!=0):
			result1_2[i][j][:] = 255

printPoints(keyPointImage1_1)
printPoints(keyPointImage1_2)
			
cv2.imwrite('octave1_1.jpg', result1_1)
cv2.imwrite('octave1_2.jpg', result1_2)

imageOctave2 = imageResize(imageOctave1)
imageColor2 = imageResizeColor(imageColor)
result2_1 = np.array(imageColor2)
result2_2 = np.array(imageColor2)
gausianKernelFliped1 = developGausian(math.sqrt(2))
gausianKernelFliped2 = developGausian(2)
gausianKernelFliped3 = developGausian(2*math.sqrt(2))
gausianKernelFliped4 = developGausian(4)
gausianKernelFliped5 = developGausian(4*math.sqrt(2))

imageConvoluted2_1 = convulateImage(imageOctave2,gausianKernelFliped1)
imageConvoluted2_2 = convulateImage(imageOctave2,gausianKernelFliped2)
imageConvoluted2_3 = convulateImage(imageOctave2,gausianKernelFliped3)
imageConvoluted2_4 = convulateImage(imageOctave2,gausianKernelFliped4)
imageConvoluted2_5 = convulateImage(imageOctave2,gausianKernelFliped5)

dog2_1 = imageConvoluted2_1 - imageConvoluted2_2
dog2_2 = imageConvoluted2_2 - imageConvoluted2_3
dog2_3 = imageConvoluted2_3 - imageConvoluted2_4
dog2_4 = imageConvoluted2_4 - imageConvoluted2_5

keyPointImage2_1 = np.ndarray(shape = (imageConvoluted2_1.shape[0],imageConvoluted2_1.shape[1]))
keyPointImage2_2 = np.ndarray(shape = (imageConvoluted2_1.shape[0],imageConvoluted2_1.shape[1]))

for i in range(3,imageConvoluted2_1.shape[0]-6):
	for j in range(3,imageConvoluted2_1.shape[1]-6):
		mat1 = imageChunkSmall(dog2_1,i,j)
		mat2 = imageChunkSmall(dog2_2,i,j)
		mat3 = imageChunkSmall(dog2_3,i,j)
		if(calMinMax(mat1,mat2,mat3)):
			keyPointImage2_1[i+1][j+1] = dog2_2[i+1][j+1]
		
		mat4 = imageChunkSmall(dog2_4,i,j)
		if(calMinMax(mat2,mat3,mat4)):
			keyPointImage2_2[i+1][j+1] = dog2_3[i+1][j+1]

for i in range(result2_1.shape[0]):
	for j in range(result2_1.shape[1]):
		if(keyPointImage2_1[i][j]!=0):
			result2_1[i][j][:] = 255
		if(keyPointImage2_2[i][j]!=0):
			result2_2[i][j][:] = 255


printPoints(keyPointImage2_1)
printPoints(keyPointImage2_2)

cv2.imwrite('dog2_1.jpg', dog2_1)
cv2.imwrite('dog2_2.jpg', dog2_2)
cv2.imwrite('dog2_3.jpg', dog2_3)
cv2.imwrite('dog2_4.jpg', dog2_4)
print('octave 2 row'+str(result2_1.shape[0]))
print('octave 2 colom'+str(result2_1.shape[1]))
cv2.imwrite('octave2_1.jpg', result2_1)
cv2.imwrite('octave2_2.jpg', result2_2)

imageOctave3 = imageResize(imageOctave2)
imageColor3 = imageResizeColor(imageColor2)
result3_1 = np.array(imageColor3)
result3_2 = np.array(imageColor3)
gausianKernelFliped1 = developGausian(2* math.sqrt(2))
gausianKernelFliped2 = developGausian(4)
gausianKernelFliped3 = developGausian(4*math.sqrt(2))
gausianKernelFliped4 = developGausian(8)
gausianKernelFliped5 = developGausian(8*math.sqrt(2))

imageConvoluted3_1 = convulateImage(imageOctave3,gausianKernelFliped1)
imageConvoluted3_2 = convulateImage(imageOctave3,gausianKernelFliped2)
imageConvoluted3_3 = convulateImage(imageOctave3,gausianKernelFliped3)
imageConvoluted3_4 = convulateImage(imageOctave3,gausianKernelFliped4)
imageConvoluted3_5 = convulateImage(imageOctave3,gausianKernelFliped5)

dog3_1 = imageConvoluted3_1 - imageConvoluted3_2
dog3_2 = imageConvoluted3_2 - imageConvoluted3_3
dog3_3 = imageConvoluted3_3 - imageConvoluted3_4
dog3_4 = imageConvoluted3_4 - imageConvoluted3_5

keyPointImage3_1 = np.ndarray(shape = (imageConvoluted3_1.shape[0],imageConvoluted3_1.shape[1]))
keyPointImage3_2 = np.ndarray(shape = (imageConvoluted3_1.shape[0],imageConvoluted3_1.shape[1]))

for i in range(3,imageConvoluted3_1.shape[0]-6):
	for j in range(3,imageConvoluted3_1.shape[1]-6):
		mat1 = imageChunkSmall(dog3_1,i,j)
		mat2 = imageChunkSmall(dog3_2,i,j)
		mat3 = imageChunkSmall(dog3_3,i,j)
		if(calMinMax(mat1,mat2,mat3)):
			keyPointImage3_1[i+1][j+1] = dog3_2[i+1][j+1]
		
		mat4 = imageChunkSmall(dog3_4,i,j)
		if(calMinMax(mat2,mat3,mat4)):
			keyPointImage3_2[i+1][j+1] = dog3_3[i+1][j+1]

for i in range(result3_1.shape[0]):
	for j in range(result3_1.shape[1]):
		if(keyPointImage3_1[i][j]!=0):
			result3_1[i][j] = 255
		if(keyPointImage3_2[i][j]!=0):
			result3_2[i][j] = 255


printPoints(keyPointImage3_1)
printPoints(keyPointImage3_2)
			
cv2.imwrite('dog3_1.jpg', dog3_1)
cv2.imwrite('dog3_2.jpg', dog3_2)
cv2.imwrite('dog3_3.jpg', dog3_3)
cv2.imwrite('dog3_4.jpg', dog3_4)
print('octave 2 row'+str(result3_1.shape[0]))
print('octave 2 colom'+str(result3_1.shape[1]))
cv2.imwrite('octave3_1.jpg', result3_1)
cv2.imwrite('octave3_2.jpg', result3_2)


#imageOctave4 = imageResize(imageOctave3)
imageOctave4 = imageResize(imageOctave3)
imageColor4 = imageResizeColor(imageColor3)
result4_1 = np.array(imageColor4)
result4_2 = np.array(imageColor4)
gausianKernelFliped1 = developGausian(2* math.sqrt(2))
gausianKernelFliped2 = developGausian(4)
gausianKernelFliped3 = developGausian(4*math.sqrt(2))
gausianKernelFliped4 = developGausian(8)
gausianKernelFliped5 = developGausian(8*math.sqrt(2))

imageConvoluted4_1 = convulateImage(imageOctave4,gausianKernelFliped1)
imageConvoluted4_2 = convulateImage(imageOctave4,gausianKernelFliped2)
imageConvoluted4_3 = convulateImage(imageOctave4,gausianKernelFliped3)
imageConvoluted4_4 = convulateImage(imageOctave4,gausianKernelFliped4)
imageConvoluted4_5 = convulateImage(imageOctave4,gausianKernelFliped5)

dog4_1 = imageConvoluted4_1 - imageConvoluted4_2
dog4_2 = imageConvoluted4_2 - imageConvoluted4_3
dog4_3 = imageConvoluted4_3 - imageConvoluted4_4
dog4_4 = imageConvoluted4_4 - imageConvoluted4_5

keyPointImage4_1 = np.ndarray(shape = (imageConvoluted4_1.shape[0],imageConvoluted4_1.shape[1]))
keyPointImage4_2 = np.ndarray(shape = (imageConvoluted4_1.shape[0],imageConvoluted4_1.shape[1]))

for i in range(3,imageConvoluted4_1.shape[0]-6):
	for j in range(3,imageConvoluted4_1.shape[1]-6):
		mat1 = imageChunkSmall(dog4_1,i,j)
		mat2 = imageChunkSmall(dog4_2,i,j)
		mat3 = imageChunkSmall(dog4_3,i,j)
		if(calMinMax(mat1,mat2,mat3)):
			keyPointImage4_1[i+1][j+1] = dog4_2[i+1][j+1]
		
		mat4 = imageChunkSmall(dog4_4,i,j)
		if(calMinMax(mat2,mat3,mat4)):
			keyPointImage4_2[i+1][j+1] = dog4_3[i+1][j+1]

for i in range(result4_1.shape[0]):
	for j in range(result4_1.shape[1]):
		if(keyPointImage4_1[i][j]!=0):
			result4_1[i][j] = 255
		if(keyPointImage4_2[i][j]!=0):
			result4_2[i][j] = 255


printPoints(keyPointImage4_1)
printPoints(keyPointImage4_2)
			
cv2.imwrite('dog4_1.jpg', dog4_1)
cv2.imwrite('dog4_2.jpg', dog4_2)
cv2.imwrite('dog4_3.jpg', dog4_3)
cv2.imwrite('dog4_4.jpg', dog4_4)
cv2.imwrite('octave4_1.jpg', result4_1)
cv2.imwrite('octave4_2.jpg', result4_2)

print('end')
cv2.waitKey(0)

