import cv2
import numpy as np
import customtkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model

class EmotionClassifier:
    def __init__(self):
        self.model = load_model('model_file_30epochs.h5')
        
        self.labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.video = None

    def classify_emotion_from_image(self, image_path):
        frame = cv2.imread(image_path)
        self.classify_emotion(frame)

    def classify_emotion_from_webcam(self):
        self.video = cv2.VideoCapture(0)
        while True:
            ret, frame = self.video.read()
            self.classify_emotion(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop_webcam()

    def stop_webcam(self):
        if self.video is not None:
            self.video.release()
            cv2.destroyAllWindows()

    def classify_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceDetect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = self.model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(label)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, self.labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

class EmotionClassifierApp:
    def __init__(self):
        self.root = tk.CTk()
        self.root.title("Emotion Classifier")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_coordinate = (screen_width - 800) // 2
        y_coordinate = (screen_height - 800) // 2
        self.root.geometry(f"400x400+{x_coordinate}+{y_coordinate}")

        self.button_frame = tk.CTkFrame(self.root)
        self.button_frame.pack(expand=True)

        self.emotion_classifier = EmotionClassifier()

        self.image_button = tk.CTkButton(self.button_frame, text="Select Image", command=self.open_file_dialog, width=300, height=100)
        self.image_button.pack(pady=10, anchor="center")

        self.webcam_button = tk.CTkButton(self.button_frame, text="Start Webcam", command=self.emotion_classifier.classify_emotion_from_webcam, width=300, height=100)
        self.webcam_button.pack(pady=10, anchor="center")

    def open_file_dialog(self):
        self.root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.emotion_classifier.classify_emotion_from_image(self.root.filename)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EmotionClassifierApp()
    app.run()
