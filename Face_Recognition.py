# Importo le librerie standard
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# Importo le librerie di Tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# Evito errori di Out Of Memory utilizzando la potenza di calcolo e memoria della mia gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# Creo i percorsi delle 3 categorie di immagini
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# Creo le corrispettive cartelle
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# trasferisco tutte le immagini nella cartella LFW (Labelled Faces in the Wild) nella corrispettiva cartella Negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)



# Importo UUID che facilitera' la distinzione delle immagini dandogli nomi unici
import uuid



# Creo una connessione con la webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Ridimensiono la cattura della webcam a 250x250 pixels
    frame = frame[120:120+250,200:200+250, :]
    
    # Inizio a catturare le immagini Ancora (Anchor)
    if cv2.waitKey(1) & 0XFF == ord('a'):

        # Tramite UUID dono all'imagine un nome unico e la posiziono nel percorso per le immagini Anchor
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Inizio a catturare le immagini Positive (Positive)
    if cv2.waitKey(1) & 0XFF == ord('p'):

        # Tramite UUID dono all'imagine un nome unico e la posiziono nel percorso per le immagini Positive
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Mostro l'immagine appena presa a schermo
    cv2.imshow('Image Collection', frame)
    
    # Utilizzo il pulsante 'q' come mezzo per interrompere il ciclo
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Rilascio la connessione con la webcam
cap.release()

# Chiudo la finestra a schermo dell'immagine
cv2.destroyAllWindows()


#definisco la funzione di Data Augmentation per variegare i dataset, la funzione assegnera' all'immagine in input delle modifiche randomiche e creera' 9 copie modificate
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data


# Creo un ciclo che applichera' la funzione di Data Augmentation per ogni immagine in Positive, e poi inserira' le immagini modificate all'interno dello stesso path dandogli un nome unico
for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())



# Creo un ciclo che applichera' la funzione di Data Augmentation per ogni immagine in Anchor, e poi inserira' le immagini modificate all'interno dello stesso path dandogli un nome unico
for file_name in os.listdir(os.path.join(ANC_PATH)):
    img_path = os.path.join(ANC_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())



# Creo dei dataset da 3000 immagini ciascuno che serviranno al training e al testing
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)



#definisco una funzione di preprocessing che servira' a ridimensionare le immagini 
def preprocess(file_path):
    
    # Prendo l'immagine dal corrispettivo percorso
    byte_img = tf.io.read_file(file_path)

    # Decodifico l'immagine in maniera che Tensorflow possa lavorarci
    img = tf.io.decode_jpeg(byte_img)
    
    # Ridimensiono l'immagine a 100x100x3
    img = tf.image.resize(img, (100,100))

    # Siccome il framework di Tensorflow usa una scala RGB 0-1 divido l'immagine per 255
    img = img / 255.0
    
    # Restituisco l'immagine preprocessata in output
    return img


# Creo un Labelled Dataset composto da un array di 1 o di 0 in base alla combinazione Anchor-Positive, Anchor-Negative
# Positives sara' un dataset composto da un array di 3000 1
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))

# Negatives sara' un dataset composto da un array di 3000 0
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

#Concateno i due dataset, un elemento di data sara' composto quindi in questa forma esempio (path di un Anchor, path di una positive/negative ,1/0)
data = positives.concatenate(negatives)



#definisco una funzione che preso in input un elemento di Data processi le due immagini dell'elemento con la funzione Preprocess che abbiamo definito prima
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# Costruisco una pipeline di dataloading
# Applico la funzione di preprocessing su tutti gli elementi di Data
data = data.map(preprocess_twin)
data = data.cache()
# mischio gli elementi di Data in modo che non sappiamo se il prossimo elemento sara' uno negativo o positivo
data = data.shuffle(buffer_size=1024)


# Partizione di Training
# Prendo il 70% del dataset
train_data = data.take(round(len(data)*.7))

#divido il dataset di training in batch da 16 immagini
train_data = train_data.batch(16)

# Evito bottleneck processando 8 immagini in piu' del batch precendente
train_data = train_data.prefetch(8)


# Partizione di Testing
# Prendo il restante 30%
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))

#indentico a prima
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)



# Definisco la funzione di embedding che ritornera' in output il modello contenente le varie Layer che verranno utilizzate
def make_embedding(): 
    # La layer di embedding inizia prendendo in input un'immagine da 100x100x3
    inp = Input(shape=(100,100,3), name='input_image')
    
    # Il primo blocco e' composto da una Convolutional Layer con 64 filtri e una dimensione da 10x10 e da una MaxPooling Layer a 64 filtri e dimensione da 2x2
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Il secondo blocco della Layer di embedding e' composto da un'altra Convolutional Layer e da un'altra Maxpooling Layer ma che manipoleranno diversamente i dati
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Il terzo blocco e' composto da una Convolutional Layer e da una MaxPooling Layer
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Il blocco finale e' composto da una convolutional a 256 filtri e dimensione 4x4, riporto tutto ad una singola dimensione con Flatten per poi utilizzando Dense portare tutto alla forma di 4096 unita'
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')



embedding = make_embedding()


# Creiamo la distance layer che sara' la Layer finale del nostro modello, posta subito dopo la Layer di Embedding
class L1Dist(Layer):
    
    # metodo init
    def __init__(self, **kwargs):
        super().__init__()
       
    # La funzione call prende in input un immagine da validare a cui poi sottrae un'immagine che puo' essere sia positive sia negative e ritornera' il valore assoluto della sottrazione
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)





def make_siamese_model(): 
    
    # Definisco l'immagine Anchor che viene data in input
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Definisco l'immagine di validazione che viene data in input
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Assegno la distance layer e la rinomino dopodiche' faccio passare le due immagini in input attraverso la Layer di Embedding
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Compongo la Layer di Classificazione che unira' le due immagini appena processate in una layer finale 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()


# definisco perdite
binary_cross_loss = tf.losses.BinaryCrossentropy()

#definisco l'optimizer
opt = tf.keras.optimizers.Adam(1e-4) 

# costruisco un percorso dove verranno salvati dei checkpoint dell'addestramento
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)



# definisco la funzione come una @tf.function poiche' in questo modo tensorflow mi offre di trasformare la mia funzione in un grafo di tensorflow, e quindi mi dona una maggiore efficienza di training
@tf.function
def train_step(batch):
    
    # GradientTape salva tutte le operazioni che avvengono nella rete neurale per effettuare una differenzazione automatica
    with tf.GradientTape() as tape:     
        # prendo le immagini anchor, positive o negative
        X = batch[:2]
        # prendo la label (1/0)
        y = batch[2]
        
        # Passiamo dei dati al modello
        yhat = siamese_model(X, training=True)

        # Ccalcolo le perdite
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calcolo i gradienti del batch di dati corrente
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # L'optimizer applica i gradienti ottenuti da gradientTape per questo batch di dati
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Ritorna le perdite
    return loss


#costruisco il training Loop
def train(data, EPOCHS):
    # Addestro la rete per X epoche
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # per ogni bacth di dati applico a quel batch la funzione train_step che applichera' i gradienti di gradeintTape
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx+1)
        
        # Salvo il checkpoint corrente
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50

#addestro la rete
train(train_data, EPOCHS)


# Salvo il modello addestrato
siamese_model.save('siamesemodel.h5')



