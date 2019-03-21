# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os, json
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from tqdm import tqdm

import pycocotools.mask as mask_utils
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

CLASS_TYPE_CONVERSION = {
  'person':     'person',
  'rider':      'rider',
  'car':        'car',
  'bus':        'bus',
  'truck':      'truck',
  'bike':       'bike',
  'motor':      'motor',
  'traffic light': 'traffic light',
  'traffic sign':  'traffic sign',
  'train':      'train'
}

TYPE_ID_CONVERSION = {
  'person':     1,
  'rider':      2,
  'car':        3,
  'bus':        4,
  'truck':      5,
  'bike':       6,
  'motor':      7,
  'traffic light': 8,
  'traffic sign':  9,
  'train':      10
}

class BDD100KSegDataset(Dataset):
    """ BDD100k Dataset: https://bair.berkeley.edu/blog/2018/05/30/bdd/
    """
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(BDD100KSegDataset, self).__init__()

        # TODO: filter images without detection annotations

        self.transforms = transforms
        self.image_dir = root
        self.label_dir = ann_file

        label_files = sorted([f for f in os.listdir(self.label_dir) if '.json' in f])
        self.id_to_img_map = [f.replace('.json', '') for f in label_files]
        self.contiguous_category_id_to_json_id = {v: v for _, v in TYPE_ID_CONVERSION.items()}
        self.labels = []
        print("Initializing dataloader")
        for label_file in tqdm(label_files):
            with open(os.path.join(self.label_dir, label_file), 'r') as f:
                label = json.load(f)

                # filter labels
                objects = label['frames'][0]['objects']
                if len(objects) < 1:
                    continue

                processed_objects = []
                for obj in objects:
                    # filter labels
                    if obj['category'] not in CLASS_TYPE_CONVERSION.keys():
                        continue

                    # process segmentation mask
                    segments = obj['segments2d']
                    if len(segments) < 1:
                        continue
                    processed_segments = []
                    for polygon in segments:
                        points = []
                        for point in polygon:
                            points += point[:-1]
                            # Bezier Curve is not yet supported
                            assert point[-1] == 'L'
                        if len(points) <= 4:
                            continue
                        processed_segments.append(points)

                    if len(processed_segments) < 1:
                        continue
                    obj['segments2d'] = processed_segments
                    processed_objects.append(obj)

                if len(processed_objects) < 1:
                    continue
                label['frames'][0]['objects'] = processed_objects
                self.labels.append(label)

        self.length = len(self.labels)

    def __len__(self):
        return self.length;

    def __getitem__(self, idx):

        # annotations
        label = self.labels[idx]

        # load image
        img = Image.open(os.path.join(self.image_dir, "%s.jpg" % label['name']))
        H, W = img.height, img.width
        img = ToTensor()(img)
        boxes = []
        classes = []
        masks = []
        for obj in label['frames'][0]['objects']:
            # TODO: further filter annotations if needed

            label_type = CLASS_TYPE_CONVERSION[obj['category']]
            classes += [TYPE_ID_CONVERSION[label_type]]
            masks += [obj['segments2d']]

            rles = mask_utils.frPyObjects(obj['segments2d'], H, W)
            bbox = mask_utils.toBbox(rles)

            assert len(bbox) == 1
            assert len(bbox[0]) == 4
            boxes += bbox.tolist()[0]

        assert len(boxes) > 0
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (W, H), mode="xywh").convert("xyxy")

        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = SegmentationMask(masks, (W, H))
        target.add_field("masks", masks)
        return img, target, idx  # label['name']

    def get_img_info(self, idx):
        label = self.labels[idx]
        # load image
        img = Image.open(os.path.join(self.image_dir, "%s.jpg" % label['name']))
        return {'width': img.width, 'height': img.height}

    # Get all gt labels. Used in evaluation.
    def get_gt_labels(self):
        return self.labels

    def get_classes_ids(self):
        return TYPE_ID_CONVERSION;

    def toCOCO(self, output_filename):
        data = dict()
        data['info'] = {}  # TODO: put info here
        data['license'] = {}
        data['categories'] = [
            {'id': v, 'name': k, 'supercategory': ""}
            for k, v in TYPE_ID_CONVERSION.items()
        ]

        images = []
        annotations = []
        annotation_id = 0

        for idx in tqdm(range(self.length)):

            # annotations
            label = self.labels[idx]

            # load image
            img = Image.open(os.path.join(self.image_dir, "%s.jpg" % label['name']))
            H, W = img.height, img.width

            for obj in label['frames'][0]['objects']:
                # TODO: further filter annotations if needed

                label_type = CLASS_TYPE_CONVERSION[obj['category']]

                rles = mask_utils.frPyObjects(obj['segments2d'], H, W)
                bbox = mask_utils.toBbox(rles)
                area = mask_utils.area(rles)

                assert len(bbox) == 1
                assert len(bbox[0]) == 4
                assert len(area) == 1

                ann_info ={
                    "id" : annotation_id,
                    "image_id" : idx,
                    "category_id" : TYPE_ID_CONVERSION[label_type],
                    "segmentation" : obj['segments2d'],
                    "area" : area.tolist()[0],
                    "bbox" : bbox.tolist()[0],
                    "iscrowd" : 0,
                }
                annotations.append(ann_info)
                annotation_id += 1

            img_info = {
                "id" : idx,
                "width" : W,
                "height" : H,
                "file_name" : "%s.jpg" % label['name'],
                "license" : 0,
                "flickr_url" : "",
                "coco_url" : "",
                "date_captured" : None,
            }
            images.append(img_info)

        assert len(images) == self.length
        assert len(annotations) == annotations[-1]["id"] + 1

        data["images"] = images
        data["annotations"] = annotations

        with open(output_filename, "w") as f:
            json.dump(data, f)

