from ultralytics import YOLO
import cv2
import torch

# Load YOLOv5n6 model
model = YOLO('yolov5n6.pt')

# Set the confidence threshold and IOU
model.conf = 0.25  # confidence threshold
model.iou = 0.45  # IOU threshold
model.agnostic = False
model.multi_label = False
model.max_det = 100  # max number of detections

# Low-resolution for inference
LOW_RES = (320, 180)

def detect_and_draw_on_image(image_path):
    # Read the input image
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError("Failed to load the image. Check the image path.")

    # Create low-res copy for detection
    low_res_frame = cv2.resize(frame, LOW_RES)

    # Perform detection
    results = model(low_res_frame, verbose=False)

    # Scale bounding boxes back to original resolution
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    # Draw bounding boxes on high-res frame
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{results[0].names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Hardcoded image path
image_path = "im.jpg"

# Run detection on the specified image
if __name__ == "__main__":
    if torch.cuda.is_available():
        model.to('cuda')
        
    result_image = detect_and_draw_on_image(image_path)
    
    # Display the result
    cv2.imshow("Detection Result", result_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    # Optionally, save the result image
    output_path = "image_with_detections.jpg"
    cv2.imwrite(output_path, result_image)
