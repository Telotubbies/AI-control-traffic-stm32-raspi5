import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker  # Assuming you have a Tracker class in tracker.py
import paho.mqtt.client as mqtt  # MQTT library for publishing messages

# Load the YOLO model
model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Capture video from file
cap = cv2.VideoCapture('test_1.mp4')

# Read the class names from coco.txt
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.splitlines()  # Use splitlines() to avoid empty strings

count = 0
tracker = Tracker()

cy1 = 354
offset = 10

counter = []

# MQTT setup
broker = "192.168.110.162"  # Replace with your MQTT broker address
status_car_topic = "status/car"
process_com_topic = "process/com"

# Global flag for controlling the start of counting
start_processing = False

# Initialize the MQTT client
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(process_com_topic)  # Subscribe to the process/com topic
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    global start_processing
    if msg.topic == process_com_topic:
        if msg.payload.decode() == "Start":
            start_processing = True
            print("Started counting cars")
        elif msg.payload.decode() == "Stop":
            start_processing = False
            print("Stopped counting cars")

client.on_connect = on_connect
client.on_message = on_message
client.connect(broker, 1883, 60)
client.loop_start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Process every third frame
        continue

    # Resize the frame if necessary
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    boxes_data = results[0].boxes.data.cpu()  # Ensure data is on CPU
    px = pd.DataFrame(boxes_data.numpy()).astype("float")  # Convert to NumPy array

    detected_boxes = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, class_id = map(int, row[:6])
        class_name = class_list[class_id]
        
        # Check for 'car' in class name
        if 'car' in class_name:
            detected_boxes.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detected_boxes)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if start_processing:  # Only process when start_processing is True
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                if counter.count(obj_id) == 0:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    counter.append(obj_id)

    # Draw lines
    cv2.line(frame, (310, cy1), (651, cy1), (255, 255, 255), 2)
    cv2.putText(frame, '1line', (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    d = len(counter)
    cv2.putText(frame, f'going down: {d}', (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    print(f"Cars counted: {d}")

    # If 5 cars are counted, send a True value to the MQTT topic
    if d >= 7:
        client.publish(status_car_topic, "true")
        print("Sent 'True' to status/car topic")
        counter = []  # Reset counter after sending the message

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()

# Stop the MQTT client loop and disconnect
client.loop_stop()
client.disconnect()
