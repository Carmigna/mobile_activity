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
3. cd ~/mobile_activity/training
4. run python3 ~/mobile_activity/training/scripts/generate_dataset.py ~/mobile_activity/data/ 96 ~/mobile_activity/training/  
5. we could choose for resolution (second argument) instead of 96 either 128, 160, 192, 224 or any add 32 but 96 is least demanding while training on a single laptop GPU or even worst on few laptop CPUs
6. It is always recommended to install tensorflow GPU from source with CUDA [check this out!](https://github.com/Carmigna/tensorflow)

## Training:
0. gedit ~/mobile_activity/training/scripts/train_model.py in line119 change the 5 in "x = Dense(5,activation='softmax', name='fc' )(x)" to the number of activity classes prepared in the database (minimum 3) then save to the executable
1. run python3 ~/mobile_activity/training/scripts/train_model.py  96 1.0 30 64 5 6 activity_model None 1 0.001 0.000001 None
2. the executable arguments respectively are:
   a- image resolution
   b- mobilenetv2 parameter alpha
   c- epochs
   d- batch size
   e- patience  (how many epochs to wait before reducing a fraction of the learning rate)
   f- threads
   g- name of the model
   h- best weights file name (it is set to none because we're starting from scratch otherwise to start from a pretrained file, it should be saved in bk folder we created in mobile_activity/training repo after the last training session) 
   i- number of elements to output (in this case we have only 1 since it's a classification algorithm)
   j- initial learnin rate
   k- final minimal learning rate
   l- path to the training repo (it is defaulted with None)
3. to use tensorboard from another terminal run tensorboard --logdir="./path/to/mobile_activity/training/logs_gen" --port 7007
4. in the browser go to http://localhost:7007/ to check the graphs
   
