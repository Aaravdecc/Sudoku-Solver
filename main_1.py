import sudoku_Mathematical
from utlis import *


pathImage = "C:\\Users\\aarav_qoo\\PycharmProjects\\pythonProject1\\Sudoku_using_Opencv\\img_2.png"
model = initializePredictionModel()


img = cv2.imread(pathImage)
if img is None:
    raise IOError("Image not found")


h, w = img.shape[:2]
scale = 600 / max(h, w)
if scale < 1:
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

imgBlank = np.zeros_like(img)


imgThreshold = preProcess(img)


imgContour = img.copy()
imgBigContour = img.copy()

contours, _ = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 3)

biggest, area = biggestContour(contours)


imgWarped = imgBlank.copy()
imgDetectedDigits = imgBlank.copy()
imgSolvedDigits = imgBlank.copy()
inv_perspective = img.copy()


if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, [biggest], -1, (0, 0, 255), 10)

    side = max(
        int(np.linalg.norm(biggest[0] - biggest[1])),
        int(np.linalg.norm(biggest[1] - biggest[3]))
    )
    side = side - (side % 9)  # divisible by 9

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [side, 0], [0, side], [side, side]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(img, matrix, (side, side))
    imgWarped = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)

    boxes = splitBoxes(imgWarped)
    numbers, probabilities = getPrediction(boxes, model)

    print("\nDetected Sudoku Grid (Digit | Confidence):\n")
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            print(f"{numbers[idx]}({probabilities[idx]:.2f})", end="  ")
        print()

    imgDetectedDigits = np.zeros((side, side, 3), np.uint8)
    imgSolvedDigits = np.zeros((side, side, 3), np.uint8)

    imgDetectedDigits = displayNumbers(
        imgDetectedDigits, numbers, color=(255, 0, 255)
    )

    numbers_np = np.array(numbers)
    posArray = np.where(numbers_np > 0, 0, 1)

    board = np.array_split(numbers_np, 9)
    try:
        sudoku_Mathematical.solve(board)
    except:
        pass

    solvedNumbers = np.array(board).flatten() * posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    pts1 = np.float32([[0, 0], [side, 0], [0, side], [side, side]])
    pts2 = np.float32(biggest)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarp = cv2.warpPerspective(
        imgSolvedDigits, matrix, (img.shape[1], img.shape[0])
    )

    inv_perspective = cv2.addWeighted(img, 0.5, imgInvWarp, 1, 0)

    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

imageArray = (
    [img, imgThreshold, imgContour, imgBigContour],
    [imgWarped, imgDetectedDigits, imgSolvedDigits, inv_perspective]
)

stackedImage = stackImages(1, imageArray)

cv2.imshow("Sudoku Solver", stackedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()



