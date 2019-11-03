import cv2
import numpy as np


def has_green_pixels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_l = np.array([40, 40, 40])
    hsv_h = np.array([70, 255, 255])
    return 255 in cv2.inRange(hsv, hsv_l, hsv_h)


num = 5
dim = 15
threshold = 4
top = 96  # android notification bar in xxxhdpi
left = 0
img = cv2.imread("nonogram" + str(num) + ".png")
outImg = img

# create contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 300, apertureSize=3)
# cv2.imshow("Display window", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=500, lines=np.array([]), minLineLength=100,
                        maxLineGap=80)
xlist = []
ylist = []
for line in lines:
    for x1, y1, x2, y2 in line:
        # if there is a horizontal or vertical line
        if abs(x1 - x2) <= threshold:
            xlist.append(x1)
        elif abs(y1 - y2) <= threshold:
            ylist.append(y1)
        print(x1, x2, y1, y2)
        cv2.line(outImg, (x1, y1), (x2, y2), (0, 0, 255), 2)

xlist.sort()
ylist.sort()
print(xlist, ylist)

# find difference in pixels per grid
xdiffs = [t - s for s, t in zip(xlist, xlist[1:])]
ydiffs = [t - s for s, t in zip(ylist, ylist[1:])]
print(xdiffs, ydiffs)
temp = [s for s in xdiffs if s > threshold]
dx = max(set(temp), key=temp.count)
temp = [s for s in ydiffs if s > threshold]
dy = max(set(temp), key=temp.count)
print(dx, dy)

# find index of the borders
# assumes perfect capture rate
xcaptured = sum([1 for s in xdiffs if abs(s - dx) <= threshold])
ycaptured = sum([1 for s in ydiffs if abs(s - dy) <= threshold])
xborders = []
yborders = []
print(xcaptured, ycaptured)
index = 0
for i in range(xcaptured):
    while abs(xdiffs[index] - dx) > threshold:
        index += 1
    xborders.append(xlist[index])
    index += 1
index = 0
for i in range(ycaptured):
    while abs(ydiffs[index] - dx) > threshold:
        index += 1
    yborders.append(ylist[index])
    index += 1
print(xborders, yborders)
print(len(xborders), len(yborders))

binary = cv2.threshold(gray, 50, 250, cv2.THRESH_BINARY)[1]
cv2.imwrite("gray" + str(num) + ".png", gray)
cv2.imwrite("binary" + str(num) + ".png", binary)

for xindex in range(len(xborders)):
    x = xborders[xindex]
    y = yborders[0]

    # https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv
    cropped = binary[top:y, x:x + dx]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 2))
    dilated = cv2.dilate(cropped, kernel, iterations=9)

    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    contourBoxes = [cv2.boundingRect(c) for c in contours]
    # sort top to bottom
    (contours, contourBoxes) = zip(*sorted(zip(contours, contourBoxes), key=lambda b: b[1][1]))

    index = 0  # index from top to bottom
    ub = []  # unpaired bounds with xb, yb, wb, hb
    for c in contours:
        # get rectangle bounding contour
        [xb, yb, wb, hb] = cv2.boundingRect(c)

        crop_num = img[top + yb:top + yb + hb, x + xb: x + xb + wb]
        has_green = has_green_pixels(crop_num)
        if has_green:
            crop_num = gray[top + yb:top + yb + hb, x + xb: x + xb + wb]
            cv2.imwrite("cropped{}_top_{}_{}.png".format(num, xindex, index), crop_num)
            print(xindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
            index = index + 1
        else:
            if ub:  # exists unpaired digit
                if ub[0] < xb:
                    wb = xb + wb - ub[0]
                    xb = ub[0]
                    yb = max(ub[1], yb)
                    hb = max(ub[3], hb)
                else:
                    wb = ub[0] + ub[2] - xb
                    yb = max(ub[1], yb)
                    hb = max(ub[3], hb)
                ub = []
                crop_num = gray[top + yb:top + yb + hb, x + xb: x + xb + wb]
                cv2.imwrite("cropped{}_top_{}_{}.png".format(num, xindex, index), crop_num)
                print(xindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
                index = index + 1
            else:
                ub = [xb, yb, wb, hb]

for yindex in range(len(yborders)):
    x = xborders[0]
    y = yborders[yindex]

    cropped = binary[y:y + dy, left:x]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 2))
    dilated = cv2.dilate(cropped, kernel, iterations=9)
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contourBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, contourBoxes) = zip(*sorted(zip(contours, contourBoxes), key=lambda b: b[1][0]))  # sort left to right

    index = 0  # index from left to right
    ub = []  # unpaired bounds with xb, yb, wb, hb
    for c in contours:
        # get rectangle bounding contour
        [xb, yb, wb, hb] = cv2.boundingRect(c)

        crop_num = img[y + yb:y + yb + hb, left + xb: left + xb + wb]
        has_green = has_green_pixels(crop_num)
        if has_green:
            crop_num = gray[y + yb:y + yb + hb, left + xb: left + xb + wb]
            cv2.imwrite("cropped{}_left_{}_{}.png".format(num, yindex, index), crop_num)
            print(yindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
            index = index + 1
        else:
            if ub:  # exists unpaired digit
                wb = xb + wb - ub[0]
                xb = ub[0]
                yb = max(ub[1], yb)
                hb = max(ub[3], hb)
                ub = []
                crop_num = gray[y + yb:y + yb + hb, left + xb: left + xb + wb]
                cv2.imwrite("cropped{}_left_{}_{}.png".format(num, yindex, index), crop_num)
                print(yindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
                index = index + 1
            else:
                ub = [xb, yb, wb, hb]

# cv2.imshow("cropped", cropped)
# cv2.waitKey(0)
cv2.imwrite("houghlines" + str(num) + ".png", outImg)
