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