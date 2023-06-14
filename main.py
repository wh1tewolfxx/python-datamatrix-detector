import cv2
import numpy as np
import pylibdmtx.pylibdmtx as dmtx



class datamatrix_detector:  
    def __init__(self, template, debug):
        self.image = template
        self.debug = debug
        self.trained = False
        self.kernel_size = 5
        self.iterations = 1
        self.threshold = 1
        self.threshold_max = 255
        self.threshold_type =  cv2.THRESH_BINARY_INV
        self.contour_mode = cv2.RETR_EXTERNAL
        self.detection_contour_mode = cv2.RETR_TREE
        self.contour_approx = cv2.CHAIN_APPROX_SIMPLE
        self.max_area = -1
        self.max_contour = None
        self.ratio = 0
        self.rect = None
        self.ratio_lower_limit = 0.9
        self.ratio_upper_limit = 1.1
        self.max_area_lower_limit = 0.9
        self.max_area_upper_limit = 2.0

    class detection:
        def __init__(self, roi, rect):
            self.roi = roi
            self.rect = rect

    def train(self):
        # Threshold the template image to extract black regions
        _, threshold = cv2.threshold(self.image, self.threshold, self.threshold_max, self.threshold_type)

        # kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        # erode = cv2.erode(threshold, kernel=kernel, iterations=self.iterations)

        # Find contours in the template image
        contours, _ = cv2.findContours(threshold, self.contour_mode, self.contour_approx)
            
        # Find the contour with the maximum area in the template image
        for contour in contours:
            # Fit a minimum area rectangle to the contour
            rect = cv2.minAreaRect(contour)

            _ , (width, height), _ = rect

            # Calculate the area of the minimum area rectangle
            rect_area = width * height

            # Check if the area is larger than the previous maximum area
            if rect_area > self.max_area:
                self.max_area = rect_area
                self.max_contour = contour
                self.ratio = min(width, height) / max(width, height)
                self.rect = rect

    def show_template_image(self):
        template_colored = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw the bounding box using the minimum area rectangle
        box = cv2.boxPoints(self.rect)
        box = np.intp(box)

        cv2.drawContours(template_colored, [box], 0, (0, 0, 255), 2)

        # Display the target image with the matched connected components
        cv2.imshow("Detected Template", template_colored)
        cv2.waitKey(0)

    def validate(self):
        # Draw the bounding box using the minimum area rectangle
        box = cv2.boxPoints(self.rect)
        box = np.intp(box)

        # Get the minimum and maximum coordinates
        x_min = np.min(box[:, 0]) - 5
        y_min = np.min(box[:, 1]) - 5
        x_max = np.max(box[:, 0]) + 5
        y_max = np.max(box[:, 1]) + 5

        # Define the ROI rectangle coordinates
        roi_x = x_min
        roi_y = y_min
        roi_width = x_max - x_min
        roi_height = y_max - y_min

        roi = self.image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        
        # Decode the barcode
        data = dmtx.decode(roi)

        # Check if any barcode was found
        if len(data) > 0:
            # Print the decoded data
            print("Decoded Data Matrix barcode:")
            print(data[0].data)
            self.trained = True
        else:
            print("No barcode found.")
            self.trained = False

    def detect(self, target_image):
        # Threshold the template image to extract black regions
        _, threshold = cv2.threshold(target_image, self.threshold, self.threshold_max, self.threshold_type)

        # kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        # erode = cv2.erode(threshold, kernel=kernel, iterations=self.iterations)
        # Find contours in the template image
        contours, _ = cv2.findContours(threshold, self.detection_contour_mode, self.contour_approx)

        detections = []
        for contour in contours:
            # Fit a minimum area rectangle to the contour
            rect = cv2.minAreaRect(contour)
            _ , (width, height), _ = rect

            
            if (width != 0 and height != 0):               
                aspect_ratio = min(width, height) / max(width, height)
                
                # Calculate the area of the minimum area rectangle
                target_area = width * height

                if (self.max_area * self.max_area_lower_limit) < target_area < (self.max_area * self.max_area_upper_limit) and (aspect_ratio * self.ratio_lower_limit) < self.ratio < (aspect_ratio * self.ratio_upper_limit):
                    
                    # Draw the bounding box using the minimum area rectangle
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)

                    # Get the minimum and maximum coordinates
                    x_min = np.min(box[:, 0]) - 5
                    y_min = np.min(box[:, 1]) - 5
                    x_max = np.max(box[:, 0]) + 5
                    y_max = np.max(box[:, 1]) + 5

                    # Define the ROI rectangle coordinates
                    roi_x = x_min
                    roi_y = y_min
                    roi_width = x_max - x_min
                    roi_height = y_max - y_min

                    roi = target_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                    d = datamatrix_detector.detection(roi=roi, rect=rect)
                    detections.append(d)        
        return detections
    
    def visualize(self, image, detections):
        target_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for detection in detections:   
            # Draw the bounding box using the minimum area rectangle
            box = cv2.boxPoints(detection.rect)
            box = np.intp(box)
            # Decode the barcode
            data = dmtx.decode(detection.roi)

            # Check if any barcode was found
            if len(data) > 0:
                # Print the decoded data
                print("Decoded Data Matrix barcode:")
                print(data[0].data)
            else:
                print("No barcode found.")
            cv2.drawContours(target_colored, [box], 0, (0, 0, 255), 2)
        return target_colored

      
# Load the template image for training
template = cv2.imread('template2.png', cv2.IMREAD_GRAYSCALE)

# Load the target image to search for similar connected components
target_image = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)

test = datamatrix_detector(template, False)

test.train()
test.validate()
if test.trained:
    test.show_template_image()
else:
    print(f"Not Trained")

detections = test.detect(target_image=target_image)

vis = test.visualize(target_image, detections=detections)

# Display the target image with the matched connected components
cv2.imshow("Matched Connected Components", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()