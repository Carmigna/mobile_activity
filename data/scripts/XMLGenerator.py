#!/usr/bin/env python
import xml.etree.ElementTree as ET
import copy
import os
import shutil

import pathlib



class XMLGenerator:
    def __init__(self, path_out='./'):



        path_to_package = pathlib.Path('Pathto../data')
        #XML_dir = "XMLGenerator"
        base_file_name = "base_annotation.xml"

        path_to_scripts = os.path.join(path_to_package, "scripts")
        path_to_base_annotation = os.path.join(path_to_scripts, base_file_name)
        self.et_base = ET.parse(path_to_base_annotation)
        self.path_out = str(path_out)+'dataset_gen_annotations/'

        if os.path.exists(self.path_out):
            shutil.rmtree(self.path_out)

        os.makedirs(self.path_out)

    
    def generate(self, name='name', label = 'label', filename='filename',  camera_width=640, camera_height=480):
        et = copy.deepcopy(self.et_base)
        root = et.getroot()
        root.find('name').text = name
        root.find('label').text = label
        root.find('filename').text = filename
        size_elem = root.find('size')
        width_elem = ET.SubElement(size_elem, "width")
        width_elem.text = str(camera_width)
        height_elem = ET.SubElement(size_elem, "height")
        height_elem.text = str(camera_height)


        et.write('{}{}.xml'.format(self.path_out, os.path.splitext(filename)[0]))





