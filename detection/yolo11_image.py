from ultralytics import YOLO
import cv2
import torch

# Charger le modèle YOLOv11 pré-entraîné
model = YOLO('yolo11s-seg.pt')  
# Résolution basse pour l'inférence
LOW_RES = (320, 180)

def detect_and_draw_on_image(image_path):
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError("Échec du chargement de l'image. Vérifiez le chemin de l'image.")

    results = model(frame)
    for result in results:
        result.show()
        result.save("segmented_person.png")

    low_res_frame = cv2.resize(frame, LOW_RES)

    results = model.predict(source=low_res_frame, conf=0.25, iou=0.45, verbose=False)

    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Chemin de l'image
image_path = "/home/cbanide/ProjetSemestriel_new/adversarial-perturbations/detection/image.jpg"

if __name__ == "__main__":
    if torch.cuda.is_available():
        model.to('cuda')
        
    result_image = detect_and_draw_on_image(image_path)
    
    cv2.imshow("Résultat de la détection", result_image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    output_path = "image_avec_detections.jpg"
    cv2.imwrite(output_path, result_image)
