import cv2
import pytesseract
import numpy as np

video_capture = cv2.VideoCapture(0)

template = cv2.imread("./template.jpg", cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error: Template image not found!")
    exit()

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(template, None)

h_template, w_template = template.shape

possibilities = {}

def extract_serial_number(image):
    result = image

    height, width, _ = result.shape
    cropped_image = result[int(0.6 * height):int(0.75*height), int(0.15*width):int(0.4 * width)]  # Adjust to capture the bottom-left corner

    return cropped_image

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    if des1 is None or des2 is None:
        cv2.imshow('Webcam Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            pts = np.float32([[0, 0], [w_template, 0], [w_template, h_template], [0, h_template]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            rect = np.array(dst, dtype=np.float32).reshape(4, 2)
            rect = np.array(sorted(rect, key=lambda x: x[1]))

            if rect[0][0] > rect[1][0]:
                rect[[0, 1]] = rect[[1, 0]]
            if rect[2][0] < rect[3][0]:
                rect[[2, 3]] = rect[[3, 2]]

            dst_rect = np.array([[0, 0], [w_template, 0], [w_template, h_template], [0, h_template]], dtype=np.float32)
            M_warp = cv2.getPerspectiveTransform(rect, dst_rect)

            cropped_bill = cv2.warpPerspective(frame, M_warp, (w_template, h_template))

            text = pytesseract.image_to_string(extract_serial_number(cropped_bill), lang='eng',
                                   config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            text = text.strip()

            if text and text[0].isalpha() and text[-1].isalpha() and len(text) == 10:
                print(text)
                if text in possibilities:
                    possibilities[text] += 1
                else:
                    possibilities[text] = 1

            cv2.imshow("filtered bill", extract_serial_number(cropped_bill))

    if possibilities:
        cv2.putText(frame, max(possibilities, key=possibilities.get), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Dollar Bill Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


