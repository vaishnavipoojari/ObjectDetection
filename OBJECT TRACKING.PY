import gradio as gr
import cv2
import os
import numpy as np

def process_video(input_video_path):
    # Set output video path
    output_video_path = "output.avi"
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Load YOLO model
model_cfg = "C:/Users/Lenovo/Downloads/jupyter projects/yolov4.cfg"
model_weights = "C:/Users/Lenovo/Downloads/jupyter projects/yolov4.weights"
labels_path = "C:/Users/Lenovo/Downloads/jupyter projects/coco.names.txt"

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    
    with open(labels_path, "r") as f:
        labels = f.read().strip().split("\n")

    # Detection and tracking parameters
    conf_threshold = 0.6
    nms_threshold = 0.3
    blob_size = (320, 320)

    def detect_objects(frame):
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=blob_size)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(layer_names)
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        filtered_boxes, filtered_class_ids, filtered_confidences = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_boxes.append(boxes[i])
                filtered_class_ids.append(class_ids[i])
                filtered_confidences.append(confidences[i])
        return filtered_boxes, filtered_class_ids, filtered_confidences

    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes, class_ids, confidences = detect_objects(frame)
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            (x, y, w, h) = box
            label = f"{labels[class_id]} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        output_video.write(frame)

    cap.release()
    output_video.release()
    return output_video_path

# Gradio interface function
def gradio_interface(video_file):
    output_path = process_video(video_file)  # Directly use the file path
    return output_path

# Custom CSS to enhance the interface's look
custom_css = """
    body {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        font-family: 'Arial', sans-serif;
        color: #fff;
    }
    .gradio-container {
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        background-color: rgba(255, 255, 255, 0.8);
    }
    .gradio-button {
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .gradio-button:hover {
        background: #45a049;
    }
    .gradio-video {
        border-radius: 8px;
    }
"""

# Create the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Object Tracking with YOLO",
    description="Upload a video to perform object tracking using YOLO.",
    theme="compact",  # Use compact theme for a cleaner interface
    layout="horizontal",  # Align inputs and outputs horizontally
    allow_flagging="never",  # Disable flagging feature for this case
    css=custom_css,  # Apply custom CSS
)

# Launch the interface on default port 7860
interface.launch(share=True)
