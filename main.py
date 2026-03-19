import cv2
import json
from ultralytics import YOLO

# 1. SETUP PATHS
MODEL_PATH = r'C:\ProKo\smart-retail-checkout\runs\detect\retail_assistant_quick\weights\best.pt'
ONTOLOGY_PATH = r'src\ontology.json'

# 2. LOAD ASSETS
model = YOLO(MODEL_PATH)
with open(ONTOLOGY_PATH, 'r') as f:
    ontology = json.load(f)

def run_live_checkout():
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    print("--- Starting Live Checkout (Press 'q' to quit) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # STEP 1: VISION - Predict on the current frame
        # stream=True makes it faster for live video
        results = model.predict(source=frame, conf=0.6, save=False, verbose=False)
        
        # STEP 2: ANNOTATION - Draw boxes on the frame to show on screen
        annotated_frame = results[0].plot()

        # STEP 3: LOGIC - Extract detected names and match with Ontology
        detected_items = [model.names[int(box.cls[0])] for box in results[0].boxes]
        
        current_total = 0.0
        display_text = "Cart: "

        for item in set(detected_items): # Use set to avoid double listing
            count = detected_items.count(item)
            if item in ontology:
                price = ontology[item]['price']
                current_total += (price * count)
                display_text += f"{item}(x{count}) "

        # STEP 4: VISUAL FEEDBACK - Add text overlay to the video window
        cv2.putText(annotated_frame, f"Total: ${current_total:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, display_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the live video
        cv2.imshow("Smart Retail Checkout - Live View", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_checkout()