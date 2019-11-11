#Hough transform for road detection
import cv2
import numpy as np
img = cv2.imread("rural_highway_0.jpg")
img = cv2.imread("Carretera1.jpg")
#img = cv2.imread("Carretera2.jpg")
#img = cv2.imread("Carretera3.jpg")
orig = img
img = cv2.GaussianBlur(img, (5, 5), 0)
height = img.shape[0]
width = img.shape[1]
"""region_of_interest_vertices = [
    (0, height), (0, 4*height/5),
    (width/2, height/2), (width, 4*height/5),
    (width, height)
]
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
img = region_of_interest(img, np.array([region_of_interest_vertices], np.int32))"""
# Beginning of processing
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # erode and dilation of the image (opening method)
skeleton = cv2.subtract(img, opening) # actual image - opened image
# End of processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert to gray scale
img_edg = cv2.Canny(gray, 100, 100)#detect the edges of the gray image
hgh_lines = cv2.HoughLinesP(img_edg, 1, np.pi/180, 30, maxLineGap = 5, minLineLength = 30)
if hgh_lines is not None:
    for line in hgh_lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1-y2) > 10:
            #if abs(x1-x2)> 1:
            cv2.line(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
print(hgh_lines)
cv2.imshow("orig", orig)
cv2.imshow("Image", img)
cv2.imshow("Edges", img_edg)
cv2.waitKey(0)
cv2.destroyAllWindows()