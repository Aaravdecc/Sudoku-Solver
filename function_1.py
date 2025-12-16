import cv2
import numpy as np
from tensorflow.keras.models import load_model

def initializePredictionModel():
    model = load_model("C:\\Users\\aarav_qoo\\PycharmProjects\\pythonProject1\\Sudoku_using_Opencv\\digit_model.h5")
    return model

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, 1, 1, 11, 2
    )
    return imgThreshold

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    if rowsAvailable:
        h, w = imgArray[0][0].shape[:2]

        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (w, h))
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR
                    )

        hor = [np.hstack(row) for row in imgArray]
        ver = np.vstack(hor)

    else:
        for i in range(rows):
            imgArray[i] = cv2.resize(
                imgArray[i], (0, 0), None, scale, scale
            )
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(
                    imgArray[i], cv2.COLOR_GRAY2BGR
                )
        ver = np.hstack(imgArray)

    return ver

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

def splitBoxes(img):
    h, w = img.shape
    h -= h % 9
    w -= w % 9
    img = img[:h, :w]

    rows = np.vsplit(img, 9)
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)

    return boxes

def displayNumbers(img, numbers, color=(0, 255, 0)):
    h, w = img.shape[:2]
    secW = w / 9
    secH = h / 9

    fontScale = min(secW, secH) / 40
    thickness = max(1, int(min(secW, secH) / 25))

    for y in range(9):
        for x in range(9):
            value = numbers[y * 9 + x]
            if value != 0:
                text = str(value)
                (tw, th), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    thickness
                )

                textX = int(x * secW + (secW - tw) / 2)
                textY = int(y * secH + (secH + th) / 2)

                cv2.putText(
                    img,
                    text,
                    (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )
    return img

def drawGrid(img):
    secW = img.shape[1] / 9
    secH = img.shape[0] / 9

    for i in range(9):
        cv2.line(img, (0, int(secH * i)),
                 (img.shape[1], int(secH * i)), (255, 255, 0), 2)
        cv2.line(img, (int(secW * i), 0),
                 (int(secW * i), img.shape[0]), (255, 255, 0), 2)

    return img

def getPrediction(boxes, model):
    digits = [0] * len(boxes)
    probs = [0.0] * len(boxes)

    imgs = []
    cell_indices = []

    for idx, image in enumerate(boxes):
        img = np.asarray(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        margin = int(min(h, w) * 0.06)
        img = img[margin:h - margin, margin:w - margin]

        block = max(11, (min(h, w) // 8) | 1)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, 2
        )

        if cv2.countNonZero(img) < 0.03 * img.size:
            continue

        coords = cv2.findNonZero(img)
        if coords is None:
            continue

        x, y, bw, bh = cv2.boundingRect(coords)
        img = img[y:y + bh, x:x + bw]

        size = max(bw, bh)
        square = np.zeros((size, size), dtype=np.uint8)
        square[
            (size - bh) // 2:(size - bh) // 2 + bh,
            (size - bw) // 2:(size - bw) // 2 + bw
        ] = img

        img = cv2.resize(square, (28, 28))
        img = img / 255.0
        img = img.reshape(28, 28, 1)

        imgs.append(img)
        cell_indices.append(idx)

    if len(imgs) == 0:
        return digits, probs

    imgs = np.array(imgs)
    predictions = model.predict(imgs, verbose=0)

    for i, pred in enumerate(predictions):
        classIndex = np.argmax(pred)
        probabilityValue = np.max(pred)
        cell = cell_indices[i]

        img = imgs[i].reshape(28, 28)
        ys, xs = np.where(img > 0.1)

        if len(xs) > 0:
            aspect_ratio = (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)
            if classIndex == 7 and aspect_ratio > 3.0:
                digits[cell] = 1
                probs[cell] = probabilityValue
                continue

        if probabilityValue > 0.5:
            digits[cell] = classIndex
            probs[cell] = probabilityValue

    return digits, probs
