import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from ultralytics import YOLO
import gradio as gr

# Configuration pour enregistrer les masques
OUTPUT_MASK_DIR = "masks"
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# YOLO Model Configuration
model = YOLO('yolo11s-seg.pt')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 100
LOW_RES = (320, 180)

# SAM Configuration
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # Replace with the path to the SAM checkpoint file
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

def detect_and_draw(frame, frame_count):
    low_res_frame = cv2.resize(frame, LOW_RES)
    results = model(low_res_frame, verbose=False)
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]
    
    for idx, detection in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{results[0].names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Extract the region of interest for SAM
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:  # Skip if ROI is empty
            continue

        # Generate masks for the ROI
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(roi_rgb)

        if masks:
            # Use the largest mask (by area)
            largest_mask = max(masks, key=lambda x: x["area"])["segmentation"]
            mask_filename = os.path.join(OUTPUT_MASK_DIR, f"mask_{frame_count}_{idx}.png")

            # Resize mask to original bounding box size
            resized_mask = cv2.resize(largest_mask.astype(np.uint8) * 255, (x2 - x1, y2 - y1))
            full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = resized_mask

            # Save the mask
            cv2.imwrite(mask_filename, full_mask)
            print(f"Saved mask: {mask_filename}")

    return frame

def process_stream():
    cap = cv2.VideoCapture("husky.mp4")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  
            processed_frame = detect_and_draw(frame, frame_count)
            yield cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    cap.release()

iface = gr.Interface(
    fn=process_stream,
    inputs=None,
    outputs="image",
    live=True,
    title="Fast Real-time Object Detection with High-Res Output",
    description="Live stream processed with YOLOv5n6 on low-res frames, results drawn on high-res frames."
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        model.to('cuda')
    iface.launch()


