from bdd100k_seg import BDD100KSegDataset

data_paths = {
    "bdd_seg_train": {
        "img_dir": "bdd100k/seg/images/train",
        "ann_file": "bdd100k/labels/seg_labels/train"
    },
    "bdd_seg_val": {
        "img_dir": "bdd100k/seg/images/val",
        "ann_file": "bdd100k/labels/seg_labels/val"
    },
    "bdd_seg_test": {
        "img_dir": "bdd100k/seg/images/test",
        "ann_file": "bdd100k/labels/seg_labels/test"
    }
}

for dataset_name, data_path in data_paths.items():
    dataset = BDD100KSegDataset(data_path['ann_file'], data_path['img_dir'], True)
    dataset.toCOCO(dataset_name + ".json")

