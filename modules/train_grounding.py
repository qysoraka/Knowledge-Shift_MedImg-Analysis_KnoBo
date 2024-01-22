
import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

random.seed(0)

def get_mapping(modality, annotations, concept, metadata, features, max_examples, train_samples):

    try:
        positive = annotations[concept]["positive"]
        negative = annotations[concept]["negative"]
    except:
        print("The concept does not exist in the JSON annotations. Try to load from the annotation folder.")
        annotation_path = f"./data/concept_annotation_{modality}/annotations_t5/{concept}"
        
        if not os.path.exists(annotation_path):
            print(f"Annotation for {concept} does not exist. Skipping ...")
            return False

        positive = []
        negative = []
        for file in os.listdir(annotation_path):
            report_id = file.split(".")[0]
            with open(f"{annotation_path}/{file}", "r") as f:
                answer = f.read().strip()
            if answer == "yes": positive.append(report_id)
            elif answer == "no": negative.append(report_id)

    positive = annotations[concept]["positive"]
    negative = annotations[concept]["negative"]
    
    positive_images = []
    negative_images = []

    if modality == "xray":
        for report_id in positive:
            images = metadata[report_id]["images"]
            for image, image_type in images:
                if image_type in ["AP", "PA"] and image in features:
                    positive_images.append(image)
        
        for report_id in negative:
            images = metadata[report_id]["images"]
            for image, image_type in images:
                if image_type in ["AP", "PA"] and image in features:
                    negative_images.append(image)
    
    elif modality == "skin":
        for report_id in positive:
            images = metadata[report_id]["images"]
            for image in images:
                if image in features:
                    positive_images.append(image)
        
        for report_id in negative:
            images = metadata[report_id]["images"]
            for image in images:
                if image in features:
                    negative_images.append(image)

    random.seed(0)
    random.shuffle(positive_images)
    random.shuffle(negative_images)

    # equally add positive and negative examples up to max_examples
    if len(positive_images) > len(negative_images):
        negative_images_selected = negative_images[:min(len(negative_images), max_examples//2)]
        positive_images_selected = positive_images[:max_examples - len(negative_images_selected)]
    else:
        positive_images_selected = positive_images[:min(len(positive_images), max_examples//2)]
        negative_images_selected = negative_images[:max_examples - len(positive_images_selected)]
    
    val_len = min(int(0.1*min(len(positive_images_selected), len(negative_images_selected))), 50)

    if val_len < 10:
        print(f"Test length too small for {concept}. Skipping ...")
        return False

    mapping = {'0': {}, '1': {}}
    positive_train, positive_val = train_test_split(positive_images_selected, test_size=val_len, random_state=0)
    negative_train, negative_val = train_test_split(negative_images_selected, test_size=val_len, random_state=0)

    mapping['1']['train'] = positive_train[:int(train_samples*0.5)]
    mapping['1']['val'] = positive_val
    mapping['0']['train'] = negative_train[:(train_samples - len(mapping['1']['train']))]
    mapping['0']['val'] = negative_val

    return mapping


def train_binary_model(concept, mapping, features, save_path):
    model_save_path = f"{save_path}/{concept}"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    positive_data_train, negative_data_train = mapping['1']['train'], mapping['0']['train']
    positive_data_val, negative_data_val = mapping['1']['val'], mapping['0']['val']

    # downsample to keep the training data balanced
    random.seed(0)