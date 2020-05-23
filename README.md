# mobile_activity
A keras Tensorflow code using Mobilenetv2 to recognise Robotic or Human activities

## Preparing the data:
1. cd ~/mobile_activity/data
2. add all images for one single activity into dataset_gen/images (each activity should be annotated separately)
3. cd scripts
4. fill in the full path in XMLGenerator.py line16: path_to_package = pathlib.Path('####/data')
5. run python3 ~/mobile_activity/data/scripts/create_training_material_dataset.py /full/path/to/mobile_activity/data/dataset_gen/images /full/path/to/mobile_activity/data/ '1st_activity' '0'
6. change only the name and label of activity starting with label 0 up in the last 2 arguments for other classes 
7. pay attention when repeated for other activities to save images and annotations somewhere else to avoid overwrite
8. Once all classes has been annotated bring back now all images to dataset_gen/images and all annotations to dataset_gen_annotations 

## Creating the Database:
1. gedit ~/mobile_activity/training/scripts/generate_dataset.py
2. add the number of your images in dataset_gen/images to "length" and save to the executable at line 93
3. run python3 ~/mobile_activity/training/scripts/generate_dataset.py ~/mobile_activity/data/ 96 ~/mobile_activity/training/  
4. we could choose for resolution (second argument) instead of 96 either 128, 160, 192, 224 or any add 32 but 96 is least demanding while training on a single laptop GPU or even worst on few laptop CPUs
5. It is always recommended to install tensorflow GPU from source with CUDA [check this out!](https://github.com/Carmigna/tensorflow)

##
