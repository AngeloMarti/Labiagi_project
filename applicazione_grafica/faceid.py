# Importo le librerie di Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Importo i componenti Kivy Uix
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Importo altre librerie di Kivy
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Importo le ultime librerie
import cv2
import tensorflow as tf
from keras.models import load_model
from layers import L1Dist
import os
import numpy as np

# creo L'applicazione e il layout
class CamApp(App):

    def build(self):
        # Componenti del Layout principale
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verifica", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verifica non iniziata", size_hint=(1,.1))

        # Aggiungo oggetti al layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # carico il modello addestrato
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # Connetto la webacam
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Continuo a eseguire automaticamente il codice 
    def update(self, *args):

        # Leggo il frame con opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # giro orizzontalmente e converto l'immagine ad una texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # carico l'immagine e la ridimensiono a 100x100
    def preprocess(self, file_path):
        # leggo l'immagine dal percorso
        byte_img = tf.io.read_file(file_path)
        # Decodifico l'immagine in maniera che Tensorflow possa lavorarci
        img = tf.io.decode_jpeg(byte_img)
        
        # Ridimensiono l'immagine a 100x100x3
        img = tf.image.resize(img, (100,100))
        # Siccome il framework di Tensorflow usa una scala RGB 0-1 divido l'immagine per 255
        img = img / 255.0
        
        # Ritorno l'immagine
        return img

    # costruisco la funzione di verifica
    def verify(self, *args):
        # Specifico le soglie
        detection_threshold = 0.7
        verification_threshold = 0.6

        # Catturo l'immagine dalla nostra webcam
        SAVE_PATH = os.path.join('application_data', 'input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # costruisco l'array di risultati
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Faccio le predizioni
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Setto la soglia di una predizione positiva
        detection = np.sum(np.array(results) > detection_threshold)
        
        # la Verification Threshold e' la proporzione di  predizioni positive / il totale dei dati positivi
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Setto il testo di verifica
        self.verification_label.text = 'Verificato' if verified == True else 'NON verificato'

        # Stampo i risultati
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified



if __name__ == '__main__':
    CamApp().run()