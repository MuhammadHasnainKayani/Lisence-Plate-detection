License Plate Detection with YOLOv8 + Text Extraction

Overview
This script detects license plates in images and videos using the YOLOv8 model. It processes media from an input source (live camera feed, video file, or image files) and extracts text from detected license plates using Tesseract OCR.

Simply provide the input source (camera, video file, or directory of images), and the script will automatically detect license plates and extract the text. The detected plates can be displayed in real-time with bounding boxes, and the extracted text is printed for each plate.

Setup
Install YOLOv8, OpenCV, and Tesseract:

```
pip install ultralytics opencv-python pytesseract
```

Download and install Tesseract OCR from official website and ensure the installation path is correctly set in the script:
 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  Your installed path in your pc 

You can also use Easyocr to text extraction
