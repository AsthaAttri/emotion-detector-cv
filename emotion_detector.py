import cv2
from deepface import DeepFace
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Emotion Detector... Press Q to quit.")

frame_count = 0
emotion_label = "detecting..."
all_emotions = {}
fps = 0
prev_time = time.time()

# Emotion to colour mapping (BGR)
emotion_colors = {
    'happy':    (0, 255, 128),
    'sad':      (255, 100, 50),
    'angry':    (0, 0, 255),
    'surprise': (0, 255, 255),
    'fear':     (150, 0, 200),
    'disgust':  (0, 140, 255),
    'neutral':  (200, 200, 200)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

    for (x, y, w, h) in faces:
        if frame_count % 5 == 0:
            face_crop = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                emotion_label = result[0]['dominant_emotion']
                all_emotions = result[0]['emotion']
            except:
                pass

        color = emotion_colors.get(emotion_label, (0, 255, 0))

        # Face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label background + text
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, emotion_label.upper(), (x+8, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Emotion bars on the right side
        if all_emotions:
            bar_x = 450
            bar_y = 80
            cv2.putText(frame, "Emotions:", (bar_x, bar_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            for i, (emo, score) in enumerate(sorted(all_emotions.items(), key=lambda x: -x[1])):
                bar_color = emotion_colors.get(emo, (200,200,200))
                bar_len = int(score * 1.5)
                by = bar_y + i * 28
                cv2.rectangle(frame, (bar_x, by), (bar_x + bar_len, by+16), bar_color, -1)
                cv2.putText(frame, f"{emo[:7]} {score:.0f}%", (bar_x, by+13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)

    # Top bar - title + FPS
    cv2.rectangle(frame, (0, 0), (640, 35), (30, 30, 30), -1)
    cv2.putText(frame, "Real-Time Emotion Detector", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (540, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Bottom instructions
    cv2.rectangle(frame, (0, 455), (640, 480), (30, 30, 30), -1)
    cv2.putText(frame, "Press Q to quit", (10, 472),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    frame_count += 1
    cv2.imshow("Real-Time Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Emotion Detector closed.")