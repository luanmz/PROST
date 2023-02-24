import cv2
import numpy as np

# Função para selecionar o objeto a ser rastreado
def select_object(frame):
    r = cv2.selectROI(frame)
    return (r, frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])

# Carrega o vídeo
cap = cv2.VideoCapture("IMG_2228.mp4")

scale_factor = 0.5

# Seleciona o objeto a ser rastreado
ret, frame = cap.read()
frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
roi, template = select_object(frame)

# Configura o algoritmo NCC
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Aplica o NCC
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Desenha a borda do objeto rastreado
    top_left = max_loc
    bottom_right = (top_left[0] + roi[2], top_left[1] + roi[3])
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Exibe o resultado
    cv2.imshow("Resultado", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()