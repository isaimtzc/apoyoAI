import cv2
import numpy as np
video = cv2.VideoCapture("Highway3.mp4")
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def lines_detection(img):
    img_orig = img
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = [
        (0, height), (0, 3 * height / 5),
        (width / 2, height / 2), (width, 3 * height / 5),
        (width, height)
    ]
    img = region_of_interest(img, np.array([region_of_interest_vertices], np.int32))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to gray scale
    Low = np.array([18,94,140])
    High = np.array([50,255,255])
    mask = cv2.inRange(hsv, Low, High)
    img_edg = cv2.Canny(mask, 25, 50)  # detect the edges of the gray image
    hgh_lines = cv2.HoughLinesP(img_edg, 1, np.pi/180, 30, maxLineGap = 20, minLineLength = 100)
    if hgh_lines is not None:
        for line in hgh_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img_orig, img_edg

while True:
    eov, frame = video.read()# if eov is false then the video has finished
    if eov == False:
        video = cv2.VideoCapture("Highway3.mp4")
        eov, frame = video.read()
    frame, edges = lines_detection(frame)
    cv2.imshow("edges", edges)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
video.release()
cv2.destroyAllWindows()