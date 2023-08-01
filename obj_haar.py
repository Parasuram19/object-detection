import cv2

# Load the classifier for the object you want to detect (e.g. a face)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the video stream from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frames from the camera
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the object in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected objects
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame with the detected objects
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()
