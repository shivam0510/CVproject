import cv2
import numpy as np

#read the image
imageArray = cv2.imread(r'C:/Users/Shivam Agarwal/Documents/homework stuff/cvip/task1.png',0)
rows = imageArray.shape[0]
colums = imageArray.shape[1]

#sobel matrix
horizontalSobelOp = [[1,0,-1],[2,0,-2],[1,0,-1]]
verticalSobelOp = [[-1,-2,-1],[0,0,0],[1,2,1]]
horizontalEdgeImage = np.ndarray(shape = (rows,colums))
verticalEdgeImage = np.ndarray(shape = (rows,colums))

dummyMatrix = np.ndarray(shape = (3,3))
for i in range(rows):
	for j in range(colums):
		sum=0
		sum1=0
		if(i == 0 or j == 0 or i == rows-1 or j == colums-1):
			if(i==0 and j==0):
				dummyMatrix = [[0,0,0],
							[0,imageArray[i][j],imageArray[i][j+1]],
							[0,imageArray[i+1][j],imageArray[i+1][j+1]]]
			elif(i==rows-1 and j==colums-1):
				dummyMatrix = [[imageArray[i-1][j-1],imageArray[i-1][j],0],
							[imageArray[i][j-1],imageArray[i][j],0],
							[0,0,0]]
							
			elif(i==0 and j== colums-1):
				dummyMatrix = [[0,0,0],
							[imageArray[i][j-1],imageArray[i][j],0],
							[imageArray[i+1][j-1],imageArray[i+1][j],0]]

			elif(i==rows-1 and j==0):
				dummyMatrix = [[0,imageArray[i-1][j],imageArray[i-1][j+1]],
							[0,imageArray[i][j],imageArray[i][j+1]],
							[0,0,0]]
							
			elif(i==0 and j !=0 and j != colums-1):
				dummyMatrix = [[0,0,0],
							[imageArray[i][j-1],imageArray[i][j],imageArray[i][j+1]],
							[imageArray[i+1][j-1],imageArray[i+1][j],imageArray[i+1][j+1]]]
			
			elif(j==0 and i != 0 and i!=rows-1):
				dummyMatrix = [[0,imageArray[i-1][j],imageArray[i-1][j+1]],
							[0,imageArray[i][j],imageArray[i][j+1]],
							[0,imageArray[i+1][j],imageArray[i+1][j+1]]]
			
			elif(i==rows-1 and j !=colums-1 and j!=0):
				dummyMatrix = [[imageArray[i-1][j-1],imageArray[i-1][j],imageArray[i-1][j+1]],
							[imageArray[i][j-1],imageArray[i][j],imageArray[i][j+1]],
							[0,0,0]]
			elif(i != rows-1 and i !=0 and j == colums-1):
				dummyMatrix = [[imageArray[i-1][j-1],imageArray[i-1][j],0],
							[imageArray[i][j-1],imageArray[i][j],0],
							[imageArray[i+1][j-1],imageArray[i+1][j],0]]
		else:
			dummyMatrix = [[imageArray[i-1][j-1],imageArray[i-1][j],imageArray[i-1][j+1]],
							[imageArray[i][j-1],imageArray[i][j],imageArray[i][j+1]],
							[imageArray[i+1][j-1],imageArray[i+1][j],imageArray[i+1][j+1]]]
		
		for k in range(3):
			for l in range(3):
				sum = sum + dummyMatrix[k][l]* horizontalSobelOp[k][l]
				sum1 = sum1 + dummyMatrix[k][l]* verticalSobelOp[k][l]
		
		horizontalEdgeImage[i,j] = sum
		verticalEdgeImage[i,j] = sum1

# Eliminate zero values with method 1
minH = 0
maxH = 0
minV = 0
maxV = 0
for i in range(rows):
	for j in range(colums):
		if (minH > horizontalEdgeImage[i][j]):
			minH = horizontalEdgeImage[i][j]
		if (maxH < horizontalEdgeImage[i][j]):
			maxH = horizontalEdgeImage[i][j]
		if (minV > verticalEdgeImage[i][j]):
			minV = verticalEdgeImage[i][j]
		if (maxV < verticalEdgeImage[i][j]):
			maxV = verticalEdgeImage[i][j]
			
cv2.imwrite('horizontalEdgeImage.jpg', horizontalEdgeImage)
cv2.imwrite('verticalEdgeImage.jpg', verticalEdgeImage)

pos_edge_y = (horizontalEdgeImage - minH) / (maxH - minH)
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)

# Eliminate zero values with method 1
pos_edge_x = (verticalEdgeImage - minV) / (maxV - minV)
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)

cv2.waitKey(0)
cv2.destroyAllWindows()

