import cv2

# tensorf = cv2.dnn.readNetFromTensorflow('saved_model.pb',  'graph.pbtxt')
tensorf = cv2.dnn.readNet('saved_model.pb')

img = cv2.imread('lighter.jpg')
rows, cols, channels = img.shape


tensorf.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

output = tensorf.forward()

for detection in output[0,0]:
    score = float(detection[2])
    if score > 0.2:

        left    = detection[3] * cols
        top     = detection[4] * rows
        right   = detection[5] * cols
        bottom  = detection[6] - rows


        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()
