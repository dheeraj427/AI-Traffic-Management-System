import cv2

url = "http://192.168.1.6:4747/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Camera could not open")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()