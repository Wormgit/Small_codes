
import cv2 as cv

img = cv.imread('img/Lenna.png')
print( img.shape)
print( type(img), img)
cv.imshow('pic title', img)
cv.waitKey(0)
cv.putText(img, 'Learn Python with Crossin', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
cv.imwrite('img/Lenna_new.png', img)





import numpy as np

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('img/Lenna_gray.png', img_gray)

_, img_bin = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
cv.imwrite('img/Lenna_bin.png', img_bin)

img_blur = cv.blur(img, (5, 5))
cv.imwrite('img/Lenna_blur.png', img_blur)

_, contours, _ = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img_cont = np.zeros(img_bin.shape, np.uint8)    
cv.drawContours(img_cont, contours, -1, 255, 3) 
cv.imwrite('img/Lenna_cont.png', img_cont)
