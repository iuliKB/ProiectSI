# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
#transform  import un model

processor = AutoImageProcessor.from_pretrained("dima806/traffic_sign_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/traffic_sign_detection")

import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

id2label = model.config.id2label

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break

    # Preprocess the image
    inputs = processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)

    # Get the predicted class label
    predicted_class_idx = outputs.logits.argmax(-1).item() #scoatem primul element /cel mai probabil
    label = id2label[predicted_class_idx]
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0, 255), 2)

    cv2.imshow("Trafic-signs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()