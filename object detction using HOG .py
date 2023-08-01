import cv2

# Load the HOG descriptor and SVM classifier for person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open a connection to the default camera
cap = cv2.VideoCapture(0)

# Loop over frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Detect people in the frame
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Draw rectangles around the detected people
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Person Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
