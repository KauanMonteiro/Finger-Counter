import cv2
import mediapipe as mp

hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    if not success:
        print("Failed to grab frame")
        break
    
    if img is not None:
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hands.process(frameRGB)
        handPoints = results.multi_hand_landmarks
        h, w, _ = img.shape

        total_count = 0

        if handPoints:
            for points in handPoints:
                mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)
                
                pontos = [(int(cord.x * w), int(cord.y * h)) for cord in points.landmark]

                dedos = [8, 12, 16, 20]  
                contador = 0

                if pontos[4][0] < pontos[3][0]:  
                    contador += 1

                for i in range(1, 5):
                    tip_id = dedos[i - 1]
                    pip_id = tip_id - 2
                    if pontos[tip_id][1] < pontos[pip_id][1]:  
                        contador += 1

                total_count += contador

            cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(img, str(total_count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
