# from source.nn import GestureRecognizer
import cv2
import numpy as np 
import torch

PATH='gesture_classifier.pt'

class APP():
    def __init__(self):
        global PATH
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier = torch.load(PATH)
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device)

        self.y = 60
        self.x = 60
        self.h = 450
        self. w = 450

    def predict(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.classifier(img).to('cpu')[0]
        print(torch.argmax(pred))
    
    def run(self):
        color_green = (0, 255, 0)
        color_black = (0, 0, 0)
        cam = cv2.VideoCapture(0)

        i = 1
        num = 0
        number = '?'
        while True:
            _, frame = cam.read()
            i += 1
            height, width, channels = frame.shape
            frame = cv2.resize(frame, (640, 360))[0: 500, 0: 360]
            frame = cv2.flip(frame, 1)

            crop = frame[20:320, 20:340].copy()
            crop = cv2.resize(crop, (64, 64))
            # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if i % 5 == 0:
                number = self.predict(crop) # RGB
                i = 1

            cv2.putText(frame, str(number), (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (20,20), (340,320), color_green, thickness=2, lineType=8, shift=0)

            cv2.imshow('my webcam', frame)
            cv2.imshow('cropped', cv2.resize(crop, (256, 256)))
            if cv2.waitKey(1) == 27: 
                break
                
        cv2.destroyAllWindows()
        return


# код ниже запустит приложение
if __name__ == '__main__':
    try:
        myApp = APP()
        myApp.run()
    except KeyboardInterrupt:
        print('\n\nApp was stopped.\n')
