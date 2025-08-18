import cv2
import numpy as np
import joblib
import pywt

model = joblib.load('/Users/lychee/Documents/Data Science Projects/Model/svc_pipeline_model.joblib')

class_dict = {'Ronaldo': 17, 'lionel_messi': 18, 'Chee Seng': 19, 'Karina': 20}
inv_class_dict = {v: k for k, v in class_dict.items()}

harcascade = "/Users/lychee/Documents/Data Science Projects/haarcascades/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(harcascade)

# Wavelet transform function (from training model)
def wavelet_trans(img, mode='db1', level=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255
    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_recon = pywt.waverec2(coeffs_H, mode)
    img_recon *= 255
    img_recon = np.uint8(img_recon)
    return img_recon

# Webcam access
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Tune for accuracy
threshold = 3.25

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        try:
            # Preprocess face same as training
            scaled_raw = cv2.resize(face_img, (32,32))
            img_har = wavelet_trans(face_img, 'db1', 5)
            scaled_har = cv2.resize(img_har, (32,32))
            combined_img = np.vstack((
                scaled_raw.reshape(32*32*3, 1),
                scaled_har.reshape(32*32, 1)
            )).reshape(1, -1).astype(float)

            # Predict + confidence check
            decision = model.decision_function(combined_img)
            # highest lookalike
            pred = model.predict(combined_img)[0]
            # confidence level
            confidence = np.max(decision)

            print("Pred:", inv_class_dict[pred], "Conf:", confidence)

            if confidence < threshold:
                label = f"Unknown ({confidence:.2f})"
                color = (0, 0, 255)  # red
            else:
                label = f"{inv_class_dict[pred]} ({confidence:.2f})"
                color = (0, 255, 0)  # green

        except Exception as e:
            color = (0, 0, 0)  # black
            label = "Error"

        # Draw rectangle and label with score
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
