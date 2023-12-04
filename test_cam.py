import cv2

def test_camera():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Test', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the camera test function
test_camera()
