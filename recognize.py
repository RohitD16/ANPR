# Required Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Convert Image in Gray
img_bgr = cv2.imread('car3.jpg')      #color bgr
# img_gray = cv2.imread('car1.jpg', 0) #gray
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

# Convert in Contours
filtered = cv2.bilateralFilter(img_gray, 11, 17, 17) # Noise Reduction(blur)
edged = cv2.Canny(filtered, 30, 200)                 # Contours
# cv2.imshow("Contours", cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find Closed Contours
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Rectangular Contour
location = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
    if len(approx) == 4:
        location = approx
        break

# Mask Area except License Plate (rectangular contour location)
mask = np.zeros(img_gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [location], 0, 255, -1)
new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

# Crop Masked Area
x, y = np.where(mask==255)
x1, y1 = np.min(x), np.min(y)
x2, y2 = np.max(x), np.max(y)
cropped_img = img_gray[x1:x2+1, y1:y2+1]
cv2.imshow("Cropped Number Plate", cropped_img)
cv2.waitKey(0)

# OCR
text = pytesseract.image_to_string(cropped_img)[1:-2]
print("\n" + "NUMBER PLATE: " + text + "\n")

with open('recog_num_plates.txt', 'a') as file:
    file.write(text + "\n")

cv2.destroyAllWindows()
