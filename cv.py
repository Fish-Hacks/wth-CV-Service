from ultralytics import YOLO
import cv2
import torch

# Load a model
model = YOLO('yolov8x.pt')  # load an official model
img = 'assets/chair.jpg'

# --- [FUNCTION] Normalize coordinates to image ---
def normalize_coords(img, coords):
    height, width, _ = img.shape
    coords[0] = coords[0] / width
    coords[1] = coords[1] / height
    coords[2] = coords[2] / width
    coords[3] = coords[3] / height
    return coords

# [FUNCTION] : Detect objects in an image and return a JSON object with the detected objects' information
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
        normalize = normalize_coords(img, [x1, y1, x2, y2])
        
        # Check if the class name is already in the output dictionary
        if class_name not in output:
            output[class_name] = []

        # Append the object information to the class name key in the output dictionary
        output[class_name].append({
            "probability": prob,
            "raw_coordinates": [x1, y1, x2, y2],
            "coordinates": {
                "top_left": {
                    "x": normalize[0],
                    "y": normalize[1]
                },
                "bottom_right": {
                    "x": normalize[2],
                    "y": normalize[3]
                }
            }
        })

    # --- (DEBUG) : Output OpenCV Image ---
    if output_img: 
        save_img(img, output)

    # --- Output JSON ---
    print(output)
    return output


# [DEBUG] : Check if CUDA is available
def cuda_check():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))  # Change 0 to the GPU index you want to use if you have multiple GPUs
        return True
    else:
        print("CUDA is not available. Make sure you have a compatible GPU and installed the necessary libraries.")
        return False


# --- [DEBUG] Save image with bounding boxes and labels ---
def save_img(img, output):
    image = cv2.imread(img)
    
    for class_name, objects in output.items():
        for obj in objects:
            x1, y1, x2, y2 = obj["raw_coordinates"]
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

detect(img, True)