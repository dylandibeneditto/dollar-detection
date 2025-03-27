import cv2
import numpy as np

# Open the default webcam
video_capture = cv2.VideoCapture(0)

# Load and preprocess the template (US dollar bill image)
template = cv2.imread("./template.jpg", cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error: Template image not found!")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(template, None)

# Define template image dimensions
h_template, w_template = template.shape

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        continue  # Skip if frame capture fails

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the webcam frame
    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    if des1 is None or des2 is None:
        cv2.imshow('Webcam Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # Skip matching if no descriptors

    # Use BFMatcher to match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:  # Proceed only if enough matches are found
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography (transformation matrix)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Get the corners of the template
            pts = np.float32([[0, 0], [w_template, 0], [w_template, h_template], [0, h_template]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw a rectangle around the detected dollar bill
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

            # Crop the detected bill using perspective transform
            rect = np.array(dst, dtype=np.float32).reshape(4, 2)
            rect = np.array(sorted(rect, key=lambda x: x[1]))  # Sort points by y-coordinates

            # Reorder points: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            if rect[0][0] > rect[1][0]:  # Ensure correct order
                rect[[0, 1]] = rect[[1, 0]]
            if rect[2][0] < rect[3][0]:
                rect[[2, 3]] = rect[[3, 2]]

            dst_rect = np.array([[0, 0], [w_template, 0], [w_template, h_template], [0, h_template]], dtype=np.float32)
            M_warp = cv2.getPerspectiveTransform(rect, dst_rect)

            cropped_bill = cv2.warpPerspective(frame, M_warp, (w_template, h_template))

            # Display the cropped bill
            cv2.imshow('Cropped Dollar Bill', cropped_bill)

    # Show the main detection frame
    cv2.imshow('Dollar Bill Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
