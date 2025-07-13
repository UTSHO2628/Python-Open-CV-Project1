 import cv2
import numpy as np
import time

np.random.seed(20)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Initialize the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colourList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(0)  # Use 0 for the default camera

        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        startTime = time.time()

        while True:
            success, image = cap.read()
            if not success:
                print("Error: Unable to read from camera.")
                break

            currentTime = time.time()
            fps = 1 / (currentTime - startTime) if (currentTime - startTime) > 0 else 0
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.8)

            if len(bboxIdx) != 0:
                for i in bboxIdx.flatten():
                    bbox = bboxs[i]
                    classConfidence = confidences[i]
                    classLabelID = classLabelIDs[i][0] if len(classLabelIDs.shape) > 1 else classLabelIDs[i]
                    classLabel = self.classesList[classLabelID]

                    displayText = "{}: {:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox
                    classColor = self.colourList[classLabelID]

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=2)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.putText(image, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
