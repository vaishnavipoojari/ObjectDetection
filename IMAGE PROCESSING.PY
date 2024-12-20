import gradio as gr
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function for processing the image
def process_image(image, action):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    
    if action == "Display RGB Channels":
        # Split image into R, G, B channels
        channels = [image[:, :, i] for i in range(3)]  # [Red, Green, Blue]
        
        # Create subplots for each channel
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        for i, channel in enumerate(channels):
            axs[i].imshow(channel, cmap=["Reds", "Greens", "Blues"][i])
            axs[i].axis('off')
            axs[i].set_title(["Red", "Green", "Blue"][i])
        
        plt.tight_layout()
        # Save the figure as an image
        fig.savefig("rgb_channels.png")
        plt.close()  # Close the plot to free memory
        return cv2.imread("rgb_channels.png")  # Return the saved image
    
    elif action == "Apply Gaussian Blur":
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred  # Return blurred image
    
    elif action == "Generate Intensity Histogram":
        # Create the histogram plot
        plt.hist(image.ravel(), bins=256, range=(0, 256))
        plt.title("Pixel Intensity Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        # Save the histogram plot as an image
        plt.savefig("histogram.png")
        plt.close()  # Close the plot to free memory
        
        # Return the saved histogram image
        return cv2.imread("histogram.png")

    elif action == "Detect Objects and Draw Boxes":
        # Convert to grayscale for object detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use a threshold to find contours (basic object detection technique)
        _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        output_image = image.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a red box

        return output_image  # Return the image with bounding boxes

# Gradio interface
interface1 = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Your Image"),
        gr.Radio(
            ["Display RGB Channels", "Apply Gaussian Blur", "Generate Intensity Histogram", "Detect Objects and Draw Boxes"], 
            label="Choose Action"
        )
    ],
    outputs=gr.Image(type="numpy", label="Processed Image"),
    title="OpenCV Image Processing Tools",
    description="Choose an action to process your image. View RGB channels, apply blur, generate intensity histogram, or detect objects in the image.",
    theme="dark",  # Enable dark mode theme
    css="""
        .gradio-container {
            background-color: #ffebcd;  /* Peach background color */
        }
        .gradio-button {
            background-color: #444;
            color: #fff;
        }
        .gradio-input {
            background-color: #333;
            color: #fff;
        }
    """
)

# Launch the interface
interface1.launch()
