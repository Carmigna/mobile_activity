#!/usr/bin/env python

import sys

import shutil

import pathlib
import os
import datetime
import time
import numpy


from XMLGenerator import XMLGenerator


import cv2

def dataset_generator(dataset_directory='./', full_path_to_output='./',name= '',label= ''):
        


    # XML generator :
    xml_generator_obj = XMLGenerator(path_out=full_path_to_output)

    for filename in os.listdir(dataset_directory):


        xml_generator_obj.generate(name = name,
                                   label = label,
                                   filename=filename,

                                   camera_width="640",
                                   camera_height="480")





if __name__ == "__main__":

    dataset_directory = sys.argv[1]
    path_to_output = sys.argv[2]
    name = sys.argv[3]
    label = sys.argv[4]
    print (dataset_directory)
    print (path_to_output)
    print (name)
    print (label)
   
    dataset_generator(dataset_directory = dataset_directory, full_path_to_output= path_to_output, name = name, label = label)
        

    print("Program test END")
