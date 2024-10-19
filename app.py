import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('plastic_classifier_with_labels.h5')

# Define image size (same size as used during training)
IMAGE_SIZE = (224, 224)

# Define class names for plastic detection
class_names = ['No Plastic', 'Plastic']

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    resized_frame = resized_frame / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=0)
    return resized_frame

def predict_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class, confidence

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        predicted_class, confidence = predict_frame(frame)

        # Define a confidence threshold
        confidence_threshold = 0.75

        if confidence >= confidence_threshold:
            if predicted_class < len(class_names):
                label = f"{class_names[predicted_class]} ({confidence * 100:.2f}%)"
                if class_names[predicted_class] == 'Plastic':
                    # Draw a smaller bounding box around the detected plastic
                    box_color = (0, 255, 0)  # Green for plastic
                    box_thickness = 2
                    top_left = (100, 100)
                    bottom_right = (frame.shape[1] - 100, frame.shape[0] - 100)
                    cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)
            else:
                label = "Unknown Class"
        else:
            label = f"No Plastic Detected ({confidence * 100:.2f}%)"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
