import cv2

harcascade = "/Users/lychee/Documents/Data Science Projects/haarcascades/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

facecascade = cv2.CascadeClassifier(harcascade)

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()