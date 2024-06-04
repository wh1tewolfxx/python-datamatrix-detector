# Datamatrix Detector

This datamatrix detector was implemented using OpenCV and pylibdmtx. The detector utilizes a training image or template 
of a datamatrix to train the detector and set the parameters. Once trained, the detector is then used to detect
possible datamatrix's. Each possible datamatrix is decoded by pylibdmtx, and a border is placed 
around successfully decoded datamatrix's regardless of orientation.

## Getting Started

Copy the dmtx_detector.py module to your app and import the python module.

### Dependencies

* OpenCV
* Numpy
* pylibdmtx

### Installation
Copy the dmtx_detector.py module to your app and import the python module. Import the DatamatrixDetector Class and visualize function.
```
from dmtx_detector import DatamatrixDetector, visualize
```

## Usage

Example usage found in the test_module.py file.

```
import cv2
from dmtx_detector import DatamatrixDetector, visualize
# Load the template image for training
template_image = cv2.imread('template_2.png', cv2.IMREAD_GRAYSCALE)

# Load the target image to search for similar connected components
target_image = cv2.imread('test_2.png', cv2.IMREAD_GRAYSCALE)

detector = DatamatrixDetector(template_image, False)

detector.train()
detector.validate()
if detector.trained:
    detector.show_template_image()
else:
    print(f"Not Trained")

detections = detector.detect(target_image=target_image)

vis = visualize(target_image, detections=detections)

# Display the target image with the matched connected components
cv2.imshow("Matched Connected Components", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Train Template

![Template Image](/assets/images/template_2.png "Template Image")

## Detected Datamatrix's in Test Image

![Test Image](/assets/images/DetectedDatamatrix.png "Test Image")



