#!/usr/bin/env python

# We load Python stuff first because afterwards it will be removed to avoid error with openCV
import sys
import pathlib
import numpy as np


print("Start Module Loading...")
import csv
import cv2
import glob
import os
import xml.etree.ElementTree as ET
import shutil

SPLIT_RATIO = 0.9


print("Loaded Modules...")



def generate_database_now(path_to_source_training_package, image_size, path_to_database_training_package):
    
    print("Start main...")
    
    ######### START OF Directories Setup

    dataset_images_folder = os.path.join(path_to_source_training_package, "dataset_gen/images")
    dataset_annotations_folder = os.path.join(path_to_source_training_package, "dataset_gen_annotations")
    
    train_images_folder = os.path.join(path_to_database_training_package, "train")
    validation_images_folder = os.path.join(path_to_database_training_package, "validation")
    
    csv_folder_path = os.path.join(path_to_database_training_package, "dataset_gen_csv")
    train_csv_output_file = os.path.join(csv_folder_path, "train.csv")
    validation_csv_output_file = os.path.join(csv_folder_path, "validation.csv")

    if not os.path.exists(dataset_images_folder):
        print("Dataset not found==>"+str(dataset_images_folder)+", please run data/create_training_material_dataset.py")
        return False
    else:
        print("Trainin Images path found ==>"+str(dataset_images_folder))

    # We clean up the training folders
    if os.path.exists(train_images_folder):
        shutil.rmtree(train_images_folder)
    os.makedirs(train_images_folder)
    print("Created folder=" + str(train_images_folder))

    if os.path.exists(validation_images_folder):
        shutil.rmtree(validation_images_folder)
    os.makedirs(validation_images_folder)
    print("Created folder=" + str(validation_images_folder))

    if os.path.exists(csv_folder_path):
        shutil.rmtree(csv_folder_path)
    os.makedirs(csv_folder_path)
    print("Created folder=" + str(csv_folder_path))

    ######### END OF Directories Setup
    print("END OF Directories Setup")


    output = []

    print("Retrieving the xml files")
    xml_files = glob.glob("{}/*xml".format(dataset_annotations_folder))
    print("END Retrieving the xml files")

    print("Reading XML Anotation Files, this could take a while depending on the number of files, please be patient...")
    for i, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)

        path = os.path.join(dataset_images_folder, tree.findtext("./filename"))
        print("path from XML==>"+str(path))

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        
        class_name = tree.findtext("./name")
        label = tree.findtext("./label")

        output.append((path, width, height, class_name, label))
        
        print("{}/{}".format(i, xml_file), end="\r")

    print(str(output))
    print(np.asarray(output).shape)

    #######Dont forget to change the length!!!!!
    length = ..........
    ############################################





    print("Object elements==>"+str(length))


    ## START CSV generation
    print("Starting CSV database Generation...")
    with open(train_csv_output_file, "w") as train, open(validation_csv_output_file, "w") as validate:
        csv_train_writer = csv.writer(train, delimiter=",")
        csv_validate_writer = csv.writer(validate, delimiter=",")

        s = 0

        for c in range(length):
             path, width, height, class_name, label = output[s]
             absolute_original_path = os.path.abspath(path)
             data_list = [absolute_original_path, width, height, class_name, label]
#
        #        # We decide if it goes to train folder or to validate folder
#
             if c <= (length) * SPLIT_RATIO:
                basename = os.path.basename(data_list[0])
                train_scaled_img_path = os.path.join(train_images_folder, basename)
                data_list[0] = os.path.abspath(train_scaled_img_path)
                csv_train_writer.writerow(data_list)
             else:
                basename = os.path.basename(data_list[0])
                validate_scaled_img_path = os.path.join(validation_images_folder, basename)
                data_list[0] = os.path.abspath(validate_scaled_img_path)
                csv_validate_writer.writerow(data_list)
#
             image = cv2.imread(absolute_original_path)
#
             cv2.imwrite(data_list[0], cv2.resize(image, (image_size, image_size)))

             s += 1



    ## END CSV generation

    print("\nDone!")

    return True


def main():
    
    
    print("Generate Database...START")
    
    if len(sys.argv) < 4:
        print("usage: generate_dataset.py path_to_source_training_package image_size path_to_database_training_package")
    else:
        path_to_source_training_package = sys.argv[1]
        image_size = int(sys.argv[2])
        path_to_database_training_package = sys.argv[3]
        
        if path_to_database_training_package == "None":
            path_to_database_training_package = str(pathlib.Path.cwd())
            print("NOT Found path_to_database_training_package, getting default:"+str(path_to_database_training_package))
        else:
            print("Found path_to_database_training_package:"+str(path_to_database_training_package))
        
        print("Path to Training Original Material:"+str(path_to_source_training_package))
        print("image_size to generate database:"+str(image_size))
        print("image_size to generate database:"+str(path_to_database_training_package))
        
        generate_database_now(  path_to_source_training_package,
                                image_size,
                                path_to_database_training_package,
                                )
        
        print("Generate Database...END")


if __name__ == "__main__":
    main()
