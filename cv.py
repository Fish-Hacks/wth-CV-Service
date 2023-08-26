from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8x.pt')  # load an official model
img = 'assets/real.jpg'

def detect(img, output_img = False):
    # Run inference
    results = model.predict(img)
    result = results[0]
    output = {}

    # Draw bounding boxes and labels of detections
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        
        class_name = result.names[class_id]
        
        # Check if the class name is already in the output dictionary
        if class_name not in output:
            output[class_name] = []

        # Append the object information to the class name key in the output dictionary
        output[class_name].append({
            "probability": prob,
            "coordinates": {
                "top_left": {
                    "x": x1,
                    "y": y1
                },
                "bottom_right": {
                    "x": x2,
                    "y": y2
                }
            }
        })

    # --- (DEBUG) : Output OpenCV Image ---
    if output_img: 
        save_img(img, output)

    # --- Output JSON ---
    print(output)
    return output


# --- [DEBUG] Save image with bounding boxes and labels ---
def save_img(img, output):
    image = cv2.imread(img)
    
    for class_name, objects in output.items():
        for obj in objects:
            x1, y1, x2, y2 = obj["coordinates"]
            probability = obj["probability"]
            
            # Draw a bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add label with class name and probability
            label = f'{class_name}: {probability}'
            cv2.putText(
                image,
                text=label,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )

    # Save the image with bounding boxes and labels
    cv2.imwrite('assets/output.jpg', image)