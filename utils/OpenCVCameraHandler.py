import cv2

class OpenCVCameraHandler:

    def __init__(self, camera_id=0) -> None:
        self.__cap = cv2.VideoCapture(camera_id)
        self.__cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.__focus_step_value = 5

        self.window_name = "RetailEye"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Focus', self.window_name, 0, 100, self.setFocus)


    def setFocus(self, focus_value, camera_focus_property_key=28):
        """
        Sets focus of the camera, should be multiple of 5, each value given will be multiplied by 5
        """
        focus_value *= self.__focus_step_value
        # print("Focus Value :: ",focus_value)
        if focus_value%5==0:
            self.__cap.set(camera_focus_property_key, focus_value) 
        else:
            print(f"focus value should be multiple of 5, found {focus_value}")

    def getFrame(self):
        """
        Gets latest frame from the camera
        """
        ret, frame = self.__cap.read()
        frame = cv2.resize(frame, (1080,720))
        return ret, frame