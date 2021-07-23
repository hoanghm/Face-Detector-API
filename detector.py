import numpy as np
import cv2
from tensorflow import keras

models_folder = './trained_models'
FRONTAL_FACE_XML = models_folder + '/' + 'haarcascade_frontalface_default.xml'
IMG_ROWS, IMG_COLS = 128,128    # input size of the trained models

class MasterDetector():
    def __init__(self): 
        self.models_names = ['male', 'young', 'chubby', 'smiling', 'eyeglasses']
        self.models = [self._load_model(model_name) for model_name in self.models_names]
        self.face_classifier = cv2.CascadeClassifier(FRONTAL_FACE_XML)

    def detect_image(self,img_link):
        faces_coors, cropped_faces, img = self.get_faces(img_link)
        
        if faces_coors is not None:
            predictions = self.predict_faces(cropped_faces)
            
            predicted_labels = np.where(predictions > 0.5, True, False)
            
            # draw 1 box for each face
            colors = np.random.randint(0,256,(len(faces_coors), 3)).tolist() # 1 color for each face
            i = 0
            for x,y,w,h in faces_coors:
                p1 = (x,y)
                p2 = (x+w, y+h)
                self.draw_box(img, p1, p2, colors[i], "person" + str(i+1))
                i += 1
    
            faces_dicts = [dict(list(zip(self.models_names, list(labels))) + [('id', 'person{}'.format(i+1))]) for i,labels in enumerate(predicted_labels)]
            return img,  faces_dicts
        
        else:
            return None, None
    
    def predict_faces(self,faces):
        if faces.ndim == 3: # a single image
            faces = np.extend_dims(faces, axis=0)
            
        num_models = len(self.models)
        predictions = [None]*num_models
        
        for i in range(num_models):
            # models[i](faces) eliminates retracing and is much faster than models[i].predict(faces)
            predictions[i] = self.models[i](faces) 
            # predictions[i] is a vector of size len(faces) containting the scores of faces
        
        return np.transpose(np.array(predictions))[0] # rows are faces, columns are predicted results
        
        
        
    def get_faces(self,img_link):
        img = cv2.imread(img_link)
        gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 3)
        if len(faces) == 0:
            return None, None, img
        
        cropped_faces = []
        for x,y,w,h in faces:
            cropped = img.copy()[y:y+h,x:x+w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(cropped, (IMG_ROWS, IMG_COLS))/255.  # crop and normalize
            cropped_faces.append(cropped)
        
        return faces, np.array(cropped_faces), img
    
    def draw_box(self, img, p1, p2, color, label):
        img = cv2.rectangle(img, p1, p2, color, 3)
        
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)
        text_w, text_h = text_size
        
        cv2.rectangle(img, p1, (p1[0]+text_w+3, p1[1]-text_h-3), color, -1)
        cv2.putText(img, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                      (255,255,255), 2)
        return img
    
    def _load_model(self, model_name):
        json_link = models_folder + '/' + model_name + '_model.json'
        h5_link = models_folder + '/' + model_name + '_model.h5'
        
        # load json file
        json_file = open(json_link, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights(h5_link)
        
        # compile model
        loaded_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        
        return loaded_model