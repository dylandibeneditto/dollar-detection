import cv2
import numpy as np

def extract_serial_number(image, lower_green, upper_green):
    """ Extracts the bright green serial number from an image using HSV thresholding. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert colors so serial number is white, background is black
    result = cv2.bitwise_not(mask)

    # Apply Gaussian blur to smooth edges
    result = cv2.GaussianBlur(result, (3, 3), 0)

    # Resize to improve OCR accuracy
    result = cv2.resize(result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply adaptive thresholding for clearer text
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return result

def nothing(x):
    """ Placeholder function for trackbar. """
    pass

# Load the image (replace with your actual image path)
image = cv2.imread("template.jpg")

# Create a window for trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for HSV lower and upper bounds
cv2.createTrackbar("H Min", "Trackbars", 60, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 98, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 70, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 55, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

while True:
    # Get trackbar positions
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Update HSV bounds
    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, s_max, v_max])

    # Apply the function
    mask = extract_serial_number(image, lower_green, upper_green)

    # Display result
    cv2.imshow("Extracted Serial Number", mask)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
