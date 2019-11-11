import cv2
import numpy as np
video = cv2.VideoCapture("Highway5.mp4")
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def lines_detection(img):
    img_orig = img
    img = cv2.GaussianBlur(img, (5, 5), 0)
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = [
        (width/8, 4*height/5),
        (width / 8, height / 2),(0.75*width, height/2),
        (0.75*width, 4*height/5),
    ]
    img = region_of_interest(img, np.array([region_of_interest_vertices], np.int32))
    # Beginning of processing
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # erode and dilation of the image (opening method)
    skeleton = cv2.subtract(img, opening)  # actual image - opened image
    # End of processing
    gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_edg = cv2.Canny(gray, 100, 100)  # detect the edges of the gray image
    hgh_lines = cv2.HoughLinesP(img_edg, 1, np.pi/180, 30, maxLineGap =1, minLineLength = 30)
    print(hgh_lines)
    if hgh_lines is not None:
        for line in hgh_lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1-x2) > 1:
                if abs(y1-y2) > 20:
                    cv2.line(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return img_orig, img_edg

while True:
    eov, frame = video.read()# if eov is false then the video has finished
    if eov == False:
        video = cv2.VideoCapture("Highway5.mp4")
        eov, frame = video.read()
    frame, edges = lines_detection(frame)
    cv2.imshow("edges", edges)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
video.release()
cv2.destroyAllWindows()