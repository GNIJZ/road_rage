import cv2

import numpy as np

cap=cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20,(640,480))

while True:
    ret, frame=cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()