import os
import json
import numpy as np

WNIDs = None


def get_categories():
    """
        Return the first item of each synset of the ilsvrc2012 categories.
        Categories are lazy-loaded the first time they are needed.
    """

    global WNIDs
    if WNIDs is None:
        WNIDs = []
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cur_dir, "categories.txt")) as f:
            for line in f:
                category = line.split(" ")[0]
                WNIDs.append(category)
    return WNIDs


def get_numpy_array(x):
    if x.is_cuda:
        return x.detach().cpu().numpy()
    else:
        return x.numpy()

class ImageNetProbabilitiesTo1000ClassesMapping():
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self.categories = get_categories()

    def __call__(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)
    

def save_to_file(logits, targets, paths):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    hard_labels_file = os.path.join(cur_dir, "clip_hard_labels.json")
    soft_labels_file = os.path.join(cur_dir, "clip_soft_labels.json")
    logits = get_numpy_array(logits)

    probabilities = ImageNetProbabilitiesTo1000ClassesMapping()(logits)
    
    soft_data = []
    hard_data = []
    for prob, logit, target, path in zip(probabilities, logits, targets, paths):
        imagenet_file = path.split("/")[-1]
        soft_labels_dict = {imagenet_file: logit.tolist()}
        hard_labels_dict = {imagenet_file: prob[0]}
        soft_data.append(soft_labels_dict)
        hard_data.append(hard_labels_dict)
        

    with open(soft_labels_file, 'a+') as f1:
        for data in soft_data:
            f1.write(json.dumps(data))
            f1.write(",\n")
    
    with open(hard_labels_file, 'a+') as f2:
        for data in hard_data:
            f2.write(json.dumps(data))
            f2.write(",\n")    
       
        
        
    
    
    
    
