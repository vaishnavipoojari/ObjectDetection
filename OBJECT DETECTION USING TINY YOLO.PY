import gradio as gr
from imageai.Detection import VideoObjectDetection
import os
import shutil
import warnings

# Suppress ResourceWarning if non-critical
warnings.filterwarnings("ignore", category=ResourceWarning)

# Function for object detection
def detect_objects(video_path):
    detector = VideoObjectDetection()
    model_path = "models/tiny-yolov3.pt"

    # Check if the model file exists
    if not os.path.exists(model_path):
        return "Error: Model file not found!"

    print("Model path:", model_path)
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    # Define paths for input and output
    input_path = "input_video.mp4"
    output_path = "output_video.mp4"  # Expected output path
    final_output_path = "output_video_final.mp4"  # Renamed output path

    # Copy uploaded video to the input path
    shutil.copy(video_path, input_path)
    print("Input video path:", input_path)
    print("Output video path:", output_path)

    # Perform object detection
    try:
        detection_results = detector.detectObjectsFromVideo(
            input_file_path=input_path,
            output_file_path=output_path,
            frames_per_second=10,
            minimum_percentage_probability=20,  # Lowered for better detection
            save_detected_video=True
        )
        print("Detection results:", detection_results)
    except Exception as e:
        print("Error during detection:", str(e))
        return f"Error during detection: {str(e)}"

    # Check if the file with double extension exists
    if os.path.exists(output_path + ".mp4"):
        shutil.move(output_path + ".mp4", final_output_path)
        return final_output_path

    # Check if the regular output file exists
    if os.path.exists(output_path):
        shutil.move(output_path, final_output_path)
        return final_output_path

    return "Error: Output video was not generated!"

# Gradio interface
interface2 = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(label="Upload Your Video for Object Detection"),
    outputs=gr.File(label="Download the Detected Video"),
    title="Tiny YOLO Object Detection in Videos",
    description="Upload a video and let Tiny YOLO detect objects. The processed video will be available for download.",
    theme="compact",  # Use a compact theme for a cleaner look
    css="""
        .gradio-container {
            background-color: #ffdab9;  /* Light peach background color */
            font-family: 'Arial', sans-serif;
        }
        .gradio-button {
            background-color: #2e3b4e;
            color: #fff;
        }
        .gradio-input, .gradio-video-input {
            background-color: #2e3b4e;
            color: #fff;
        }
        .gradio-file {
            background-color: #2e3b4e;
            color: #fff;
        }
        .gradio-title {
            color: #2e3b4e;  /* Dark blue title color */
        }
        .gradio-description {
            color: #444;  /* Dark gray for description */
        }
    """
)

# Launch Gradio app
app = interface2.launch()

# Ensure cleanup on exit
try:
    print("Gradio app is running. Use the link above to test.")
    input("Press Enter to stop the app...\n")
finally:
    app.close()
    print("Gradio app closed.")
