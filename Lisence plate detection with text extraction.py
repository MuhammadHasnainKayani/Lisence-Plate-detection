######################### DatasetDownload ######################################

from roboflow import Roboflow
rf = Roboflow(api_key="8W5NJL9wOPDLDjAW2q")
project = rf.workspace("mochoye").project("license-plate-detector-ogxxg")
version = project.version(1)
dataset = version.download("yolov8")




######################### Downloading Model and training it on dataset adjust path in data.yaml according to your directory ######################################
from ultralytics import YOLO
model = YOLO('yolov8m.pt')

# Load custom dataset and train the model
data_path = r"License-Plate-Detector-1\roboflow\data.yaml"
model.train(data=data_path, epochs=100)

# Save the trained model
save_path = "Saved-Yolo-Folder/"
model.save(save_path)

##################################### Test using video or live cam ############################################################

import cv2
from ultralytics import YOLO
import pytesseract

detected_plates_list=[]
# Load the YOLOv8 model with the trained weights
model = YOLO("Yolo_finetuned_model/weights/best.pt")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Open the video file    ### you can use these lines to test it on video
# video_path = r"C:\Users\muham\Downloads\Video\2103099-uhd_3840_2160_30fps.mp4"
# cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Loop to continuously capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()
    best_text = ""
    best_confidence = 0.5

    for result in results:
        for detection, conf in zip(result.boxes.xyxy, result.boxes.conf):
            x1, y1, x2, y2 = detection

            plate_img = frame[int(y1):int(y2), int(x1):int(x2)]

            # Apply preprocessing techniques to enhance the plate image
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')

            # Check if the text is clearer than the current best text
            if conf > best_confidence:
                detected_plates_list.append(plate_text)
                print("License Plate Text:", plate_text)

    print("Best License Plate Text:", best_text)
    # Display the annotated frame
    cv2.imshow('Object Detection', annotated_frame)

    # Wait for 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

print(list)


##################################### Test using images  ############################################################

import cv2
from ultralytics import YOLO
import pytesseract

detected_plates_list = []

# Load the YOLOv8 model with the trained weights
model = YOLO("Yolo_finetuned_model/weights/best.pt")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image file
image_path = r"Input images/testimg2.png"
image = cv2.imread(image_path)

# Perform object detection on the image
results = model(image)

# Draw bounding boxes and labels on the image
annotated_image = results[0].plot()
best_text = ""
best_confidence = 0.1

for result in results:
    for detection, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = detection

        plate_img = image[int(y1):int(y2), int(x1):int(x2)]

        # Apply preprocessing techniques to enhance the plate image
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')

        # Check if the text is clearer than the current best text
        if conf > best_confidence:
            detected_plates_list.append(plate_text)
            print("License Plate Text:", plate_text)

print("Best License Plate Text:", best_text)

# Display the annotated image
cv2.imshow('Object Detection', annotated_image)

# Wait for 'q' key to quit the window
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()



##################################### Test using multiple images by iterating over directory ############################################################

import cv2
import os
from ultralytics import YOLO
import pytesseract

detected_plates_list = []

# Load the YOLOv8 model with the trained weights
model = YOLO("Yolo_finetuned_model/weights/best.pt")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory containing images
images_directory = "yolo_test_images"
# Fixed frame dimensions
frame_width = 400
frame_height = 400

# Iterate over all files in the directory
for filename in os.listdir(images_directory):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image file
        image_path = os.path.join(images_directory, filename)
        image = cv2.imread(image_path)

        # Perform object detection on the image
        results = model(image)

        # Draw bounding boxes and labels on the image
        annotated_image = results[0].plot()
        annotated_image = cv2.resize(annotated_image, (frame_width, frame_height))
        best_text = ""
        best_confidence = 0.1

        for result in results:
            for detection, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = detection

                plate_img = image[int(y1):int(y2), int(x1):int(x2)]

                # Apply preprocessing techniques to enhance the plate image
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')

                # Check if the text is clearer than the current best text
                if conf > best_confidence:
                    detected_plates_list.append(plate_text)
                    print("License Plate Text:", plate_text)
        # Display the annotated image
        cv2.imshow('Object Detection', annotated_image)

        # Wait for 'q' key to quit the window
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
