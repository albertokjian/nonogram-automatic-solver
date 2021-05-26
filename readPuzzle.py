import cv2
import numpy as np
import os
from lobe import ImageModel


def has_green_pixels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_l = np.array([40, 40, 40])
    hsv_h = np.array([70, 255, 255])
    return 255 in cv2.inRange(hsv, hsv_l, hsv_h)


def createCroppedDigits(num):
    threshold = 4
    top = 96  # android notification bar in xxxhdpi
    left = 0
    path = os.path.join(os.getcwd(), str(num))
    if os.path.exists(path):
        return

    os.mkdir(path)
    img = cv2.imread("nonogram" + str(num) + ".png")
    outImg = img

    # create contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 300, apertureSize=3)
    # cv2.imshow("Display window", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=500, lines=np.array([]), minLineLength=100, maxLineGap=80)
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
    cv2.imwrite("houghlines" + str(num) + ".png", outImg)

    # do not combine two digits, treat as separate digits for ML
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
                cv2.imwrite(os.path.join(path, 'top_{}_{}_single.png'.format(xindex, index)), crop_num)
                print(xindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
                index = index + 1
            else:
                if ub:  # exists unpaired digit
                    if ub[0] < xb:
                        (xbl, ybl, wbl, hbl) = ub
                        xbr, ybr, wbr, hbr = xb, yb, wb, hb
                    else:
                        xbl, ybl, wbl, hbl = xb, yb, wb, hb
                        (xbr, ybr, wbr, hbr) = ub
                    ub = []
                    crop_num = gray[top + ybl:top + ybl + hbl, x + xbl: x + xbl + wbl]
                    cv2.imwrite(os.path.join(path, 'top_{}_{}_left.png'.format(xindex, index)), crop_num)
                    print(xindex, index, '\t', xbl, ybl, wbl, hbl, x, y, has_green)
                    crop_num = gray[top + ybr:top + ybr + hbr, x + xbr: x + xbr + wbr]
                    cv2.imwrite(os.path.join(path, 'top_{}_{}_right.png'.format(xindex, index)), crop_num)
                    print(xindex, index, '\t', xbr, ybr, wbr, hbr, x, y, has_green)
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
        (contours, contourBoxes) = zip(
            *sorted(zip(contours, contourBoxes), key=lambda b: b[1][0]))  # sort left to right

        index = 0  # index from left to right
        ub = []  # unpaired bounds with xb, yb, wb, hb
        for c in contours:
            # get rectangle bounding contour
            [xb, yb, wb, hb] = cv2.boundingRect(c)

            crop_num = img[y + yb:y + yb + hb, left + xb: left + xb + wb]
            has_green = has_green_pixels(crop_num)
            if has_green:
                crop_num = gray[y + yb:y + yb + hb, left + xb: left + xb + wb]
                cv2.imwrite(os.path.join(path, 'left_{}_{}_single.png'.format(yindex, index)), crop_num)
                print(yindex, index, '\t', xb, yb, wb, hb, x, y, has_green)
                index = index + 1
            else:
                if ub:  # exists unpaired digit
                    (xbl, ybl, wbl, hbl) = ub
                    xbr, ybr, wbr, hbr = xb, yb, wb, hb
                    ub = []
                    crop_num = gray[y + ybl:y + ybl + hbl, left + xbl: left + xbl + wbl]
                    cv2.imwrite(os.path.join(path, 'left_{}_{}_left.png'.format(yindex, index)), crop_num)
                    print(xindex, index, '\t', xbl, ybl, wbl, hbl, x, y, has_green)
                    crop_num = gray[y + ybr:y + ybr + hbr, left + xbr: left + xbr + wbr]
                    cv2.imwrite(os.path.join(path, 'left_{}_{}_right.png'.format(yindex, index)), crop_num)
                    print(xindex, index, '\t', xbr, ybr, wbr, hbr, x, y, has_green)
                    index = index + 1
                else:
                    ub = [xb, yb, wb, hb]


def createBoardString(num):
    createCroppedDigits(num)
    path = os.path.join(os.getcwd(), str(num))
    model = ImageModel.load(os.path.join(os.getcwd(), 'model/Nonogram Label TensorFlow'))
    topDict = {}
    leftDict = {}
    for filename in os.listdir(path):
        properties = os.path.splitext(filename)[0].split('_')
        useDict = topDict if properties[0] == 'top' else leftDict
        cellIndex = properties[1]
        numberIndex = properties[2]
        result = model.predict_from_file(os.path.join(path, filename))
        prediction = int(result.prediction)
        confidence = [c for (p, c) in result.labels if p == result.prediction].pop()
        if confidence < 0.95:
            cv2.imshow(os.path.join(path, filename))
            prediction = input("Please check the image and enter the number you see:")
        prediction = prediction * 10 if properties[3] == 'left' else prediction
        if cellIndex not in useDict:
            useDict[cellIndex] = {numberIndex: prediction}
        elif numberIndex not in useDict[cellIndex]:
            useDict[cellIndex][numberIndex] = prediction
        else:
            useDict[cellIndex][numberIndex] += prediction

    puzzleFile = f"{num}.txt"
    with open(f"{num}.txt", 'w') as text_file:
        for dictionary in [leftDict, topDict]:
            array = []
            for _, numberDict in sorted(dictionary.items(), key=lambda t: int(t[0])):
                string = ""
                for _, value in sorted(numberDict.items(), key=lambda t: int(t[0])):
                    string += chr(value + ord('A') - 1)
                array.append(string)
            print(" ".join(array), file=text_file)
    return puzzleFile
