from Detector import Detector
import os

def main():
    # Path to model files
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Create Detector object
    detector = Detector(None, configPath, modelPath, classesPath)
    
    # Use the camera for detection......
    print("Starting object detection using webcam...")
    detector.onVideo()

 if __name__ == "__main__":
    main()
