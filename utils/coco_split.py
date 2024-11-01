import os
import json
import funcy
from sklearn.model_selection import train_test_split

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def train_val_split(json_path, save_root, split=0.8):

    with open(json_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        X_train, X_test = train_test_split(images, train_size=split)

        anns_train = filter_annotations(annotations, X_train)
        anns_test  = filter_annotations(annotations, X_test)
        
        train_path = os.path.join(save_root, "train.json")
        val_path = os.path.join(save_root, "val.json")
        save_coco(train_path, info, licenses, X_train, anns_train, categories)
        save_coco(val_path, info, licenses, X_test, anns_test, categories)

        print("Saved {} entries in {} and {} in {}".format(len(anns_train), "train.json", len(anns_test), "val.json"))
        
    return len(anns_train)
