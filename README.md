# mobile_activity
A keras Tensorflow code using Mobilenetv2 to recognise Robotic or Human activities

## Prepare the data:
1. cd ~/mobile_activity/data
2. add all images for one single activity into dataset_gen/images (each activity should be annotated separately)
3. cd scripts
4. change only the name and label of activity starting with label 0 up
5. fill in the full path in XMLGenerator.py line16: path_to_package = pathlib.Path('####/data')
6. run python3 ~/mobile_activity/data/scripts/create_training_material_dataset.py /full/path/to/mobile_activity/data/dataset_gen/images /full/path/to/mobile_activity/data/ '1st_activity' '0'
7. pay attention when repeated for other activities to save images and annotations somewhere else to avoid overwrite
