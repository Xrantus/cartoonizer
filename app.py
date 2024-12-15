import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

def cartoonize_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a median blur to reduce noise
    gray = cv2.medianBlur(gray, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        7, 7  # Adjusted parameters for more sensitive edges
    )

    # Apply bilateral filter to the original frame
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # Combine edges with the smoothed color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize the frame for better performance (optional)
        frame = cv2.resize(frame, (640, 480))

        # Apply cartoonization
        cartoon_frame = cartoonize_frame(frame)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', cartoon_frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Cartoonizer</h1><p>Go to <a href='/video_feed'>/video_feed</a> to see the cartoonized video stream.</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
