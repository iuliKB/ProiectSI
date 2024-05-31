import cv2
from libcamera import libcamera, libcamera_core
from libcamera import libcamera_core as core
from libcamera import controls

# Function to convert the captured frame to an OpenCV format
def convert_frame_to_opencv(frame):
    # The frame is in YUV420 format, convert it to BGR for OpenCV
    image = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    return image

# Initialize libcamera
camera_manager = libcamera.CameraManager()
camera = camera_manager.get_camera(0)

# Configure the camera
config = camera.create_still_configuration()
camera.configure(config)

# Start the camera
camera.start()

# Create a request and capture a frame
request = camera.create_request()
stream = config.at(0).stream()
buffer = camera.acquire_buffer(stream)
request.add_buffer(stream, buffer)
camera.queue_request(request)

# Allow the camera to warm up
import time
time.sleep(2)

# Capture frames continuously
while True:
    # Capture a frame
    camera.capture(request)
    frame = buffer.map()

    # Convert the frame to OpenCV format
    image = convert_frame_to_opencv(frame)

    # Resize the image to 224x224
    resized_image = cv2.resize(image, (224, 224))

    # Display the resized image
    cv2.imshow("Resized Frame", resized_image)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.stop()
cv2.destroyAllWindows()
