# from source.nn import GestureRecognizer
import cv2
import numpy as np 
import torch

PATH='gesture_classifier.pt'

class APP():
    def __init__(self):
        global PATH
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.classifier = torch.load(PATH)
            self.classifier.eval()
            self.classifier = self.classifier.to('cpu')
        except:
            print('Невозможно загрузить модель!')
            self.classifier = None
        
        self.input_size = 224
        self.crop_size = 300

    def predict(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img)
        img = img - img.min()
        img = img / img.max()
        img = img.unsqueeze(0)
        
        if self.classifier is not None:
            with torch.no_grad():
                pred = self.classifier(img).to('cpu')[0]
            return torch.argmax(pred)
        else:
            return 0
            
    
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
            frame = cv2.flip(frame, 1)
            
            
            crop = frame[60:420, 140:500].copy()
            crop = cv2.resize(crop, (224, 224))
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
