import cv2
import numpy as np
from flask import Flask, Response, render_template

app = Flask(__name__)

camera = cv2.VideoCapture(0)


# Function to detect faces in a frame
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

def replace(frame, points, replacement_image):
    for (x, y, w, h) in points:
        replacement_image = cv2.resize(replacement_image, (w, h))
        frame[y:y+h, x:x+w] = replacement_image
    return frame

def detect_eyes(frame):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return eyes



def color_intensify(img,in_b,in_g,in_r):
    b,g,r=cv2.split(img)
    increased_b = np.clip(b*in_b, 0, 255).astype(np.uint8)
    increased_g = np.clip(g*in_g, 0, 255).astype(np.uint8)
    increased_r = np.clip(r*in_r, 0, 255).astype(np.uint8)
    modified_image = cv2.merge((increased_b, increased_g, increased_r))
    return modified_image

def dogFace():
    replacement_image = cv2.imread(r"C:\Users\Lakshmi\Downloads\dog.jpg")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = detect_faces(frame)
            frame = replace(frame, faces, replacement_image)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def eye():
    replacement_image = cv2.imread(r"C:\Users\Lakshmi\Downloads\eye.jpg")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            eyes = detect_eyes(frame)
            frame = replace(frame, eyes, replacement_image)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_gray_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_red_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = color_intensify(frame,1,1,5)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_red_frames2():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = color_intensify(frame,1,5,1)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_green_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = color_intensify(frame,1,5,1)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_blue_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = color_intensify(frame,5,1,1)
            ret,buffer=cv2.imencode('.jpg',np.array(frame))
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def thresh(img,max_val,thresh,convert_type):
    _,img=cv2.threshold(img,thresh,max_val,convert_type)
    return img

def generate_binary():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = thresh(frame, 255, 100, cv2.THRESH_BINARY)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_thresholded_color():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = thresh(frame, 255, 100, cv2.THRESH_BINARY)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



#
# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # Count the number of detected faces
# num_people = len(faces)
#
# # Draw rectangles around the detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
# # Display the image with the face detections
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Print the number of people detected
# print("Number of people:", num_people)



def detect_face():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_gray_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed3():
    return Response(generate_red_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed4():
    return Response(generate_green_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed5():
    return Response(generate_blue_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed5')
def video_feed6():
    return Response(generate_binary(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed6')
def video_feed8():
    return Response(dogFace(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed7')
def video_feed9():
    return Response(eye(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
