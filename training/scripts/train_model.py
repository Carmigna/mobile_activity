#!/usr/bin/env python

# We load Python stuff first because afterwards it will be removed to avoid error with openCV
import sys
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import time
import csv
import math
import pathlib
from sklearn.metrics import classification_report
#from imutils import paths
import matplotlib.pyplot as plt

import argparse
import pickle
import cv2
import numpy as np

from keras import Model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet_v2 import _inverted_res_block
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import MaxPooling2D, Conv2D, Reshape, Flatten, Dense
from keras.utils import Sequence
from keras.optimizers import Adam
import keras.backend as K
K.set_floatx('float32')
#K.image_data_format() == 'channels_last'
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()


class DataSequence(Sequence):

    def __load_images(self, dataset):
        return np.array([cv2.imread(f,1) for f in dataset], dtype='f')

    def __init__(self, csv_file, batch_size=32, inmemory=False, number_of_elements_to_be_output=1):
        self._number_of_elements_to_be_output = number_of_elements_to_be_output
        
        assert (self._number_of_elements_to_be_output == 1)
        
        self.paths = []
        self.batch_size = batch_size
        self.inmemory = inmemory

        with open(csv_file, "r") as file:            
            self.y = np.zeros((sum(1 for line in file), ),dtype=int)
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, (scaled_img_path, _, _, _, label) in enumerate(reader):                
                if self._number_of_elements_to_be_output == 1:
                    self.y[index] = label
                    
                    
                self.paths.append(scaled_img_path)

            
            self.y = lb.fit_transform(self.y)
            print (self.y)
            print (str(self.paths))

        if self.inmemory:
            self.x = self.__load_images(self.paths)
            self.x = preprocess_input(self.x)
        #print(self.x)
    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)
    
    def __getitem__(self, idx):
        
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.inmemory:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

        batch_x = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x)
        images = preprocess_input(images)
        
        
        
        #print(batch_y)
        
        return images, batch_y

def create_model(size, alpha, number_of_elements_to_be_output):
    model = MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha)

    # to freeze layers
    # for layer in model.layers:
    #     layer.trainable = False
#
    x = model.layers[-1].output
    if size == 96:
        kernel_size_adapt = 3
    elif size == 128:
        kernel_size_adapt = 4
    elif size == 160:
        kernel_size_adapt = 5
    elif size == 192:
        kernel_size_adapt = 6
    elif size == 224:
        kernel_size_adapt = 7
    else:
        kernel_size_adapt = 1
    

    x = Flatten()(x)
    x = Dense(5,activation='softmax', name='fc' )(x)

    from keras.utils import plot_model
    plot_model(model, to_file='./model.png', show_shapes=True)
    


    return Model(inputs=model.input, outputs=x)





def train(model, epochs, batch_size, patience, threads, train_csv, validation_csv, models_weight_checkpoints_folder, logs_folder, model_unique_id, load_weight_starting_file=None, number_of_elements_to_be_output=2, initial_learning_rate=0.0001, min_learning_rate = 1e-8):
    train_datagen = DataSequence(train_csv, batch_size, False, number_of_elements_to_be_output)
    validation_datagen = DataSequence(validation_csv, batch_size, False,number_of_elements_to_be_output)

    if load_weight_starting_file:

        print("Preload Weights, to continue prior training...."+str(load_weight_starting_file))
        model.load_weights(load_weight_starting_file)
    else:

        print("Starting from empty weights.......")


    adam_optim = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy", optimizer=adam_optim, metrics=["accuracy"])

    full_model_file_name = "model-"+ model_unique_id + "-{val_loss:.8f}.h5"
    model_file_path = os.path.join(models_weight_checkpoints_folder, full_model_file_name)
    checkpoint = ModelCheckpoint(model_file_path, monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    
    stop = EarlyStopping(monitor="val_loss", patience=patience*5, mode="auto")
    
    # https://rdrr.io/cran/kerasR/man/ReduceLROnPlateau.html
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=patience, min_lr=min_learning_rate, verbose=1, mode="auto")

    tensorboard_clb = TensorBoard(log_dir=logs_folder, histogram_freq=0,
                                write_graph=True, write_images=True)

    model.summary()
    

    model.fit_generator(generator=train_datagen,
                        epochs=epochs,
                        validation_data=validation_datagen,
                        callbacks=[checkpoint, reduce_lr, stop, tensorboard_clb],
                        workers=threads,
                        use_multiprocessing=True,
                        shuffle=True,
                        verbose=1)


def main():



    print("Train Model...START")
    
    if len(sys.argv) < 13:
        
        print("usage: train_model.py image_size ALPHA EPOCHS BATCH_SIZE PATIENCE THREADS training_name weight_file_name number_of_elements_to_be_output initial_learning_rate min_learning_rate path_to_database_training_package")
    else:
        image_size = int(sys.argv[1])
        ALPHA = float(sys.argv[2])
        EPOCHS = int(sys.argv[3])
        BATCH_SIZE = int(sys.argv[4])
        PATIENCE = int(sys.argv[5])
        THREADS = int(sys.argv[6])
        training_name = sys.argv[7]
        weight_file_name = sys.argv[8]
        number_of_elements_to_be_output = int(sys.argv[9])
        initial_learning_rate = float(sys.argv[10])
        min_learning_rate = float(sys.argv[11])
        path_to_database_training_package = sys.argv[12]
        

    
        if path_to_database_training_package == "None":

            path_to_database_training_package = str(pathlib.Path.cwd())
            
        else:
            
            print("Training Databse FOUND, setting default:"+str(path_to_database_training_package))
    
        csv_folder_path = os.path.join(path_to_database_training_package, "dataset_gen_csv")
        train_csv_output_file = os.path.join(csv_folder_path, "train.csv")
        print('csv train output file path = '+str(train_csv_output_file))
        validation_csv_output_file = os.path.join(csv_folder_path, "validation.csv")
        
        models_weight_checkpoints_folder = os.path.join(path_to_database_training_package, "model_weight_checkpoints_gen")
        logs_folder = os.path.join(path_to_database_training_package, "logs_gen")
        
        # We clean up the training folders
        if os.path.exists(models_weight_checkpoints_folder):
            shutil.rmtree(models_weight_checkpoints_folder)
        os.makedirs(models_weight_checkpoints_folder)
        print("Created folder=" + str(models_weight_checkpoints_folder))
    
        if os.path.exists(logs_folder):
            shutil.rmtree(logs_folder)
        os.makedirs(logs_folder)
        print("Created folder=" + str(logs_folder))
    
        print ("Start Create Model")
        model = create_model(image_size, ALPHA, number_of_elements_to_be_output)
        
        
        model_unique_id = training_name +"-"+ str(image_size) +"-"+ str(ALPHA) +"-"+ str(EPOCHS) +"-"+ str(BATCH_SIZE) + "-TIME-" + str(time.time())
        
        
        if weight_file_name != "None":

            path_to_package = str(pathlib.Path.cwd())
            backup_models_weight_checkpoints_folder = os.path.join(path_to_package, "bk")
            load_weight_starting_file = os.path.join(backup_models_weight_checkpoints_folder, weight_file_name)
        else:
           
            print("No load_weight_starting_file...We will start from scratch!")
            load_weight_starting_file = None
        
        
        train(model,
              EPOCHS,
              BATCH_SIZE,
              PATIENCE,
              THREADS,
              train_csv_output_file,
              validation_csv_output_file,
              models_weight_checkpoints_folder,
              logs_folder,
              model_unique_id,
              load_weight_starting_file,
              number_of_elements_to_be_output,
              initial_learning_rate,
              min_learning_rate)
        
        
        print("Train Model...END")

        
        
        print("[INFO] serializing network...")
        model.save("model/activity.model")
        # serialize the label binarizer to disk
        f = open("model/lb.pickle", "wb")
        f.write(pickle.dumps(lb))
        f.close()


if __name__ == "__main__":
    main()
