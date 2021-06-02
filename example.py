import torchvision
import torch
import time
import json
from collections import defaultdict
import os
import numpy as np
import cv2

from crow_dataset import CroWTiledDataset, BaseDataset, DatasetElement, DatasetElementInfo


class COCOFormatDataset(BaseDataset, torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, max_whole_image_size,
                 use_dummy_images=False,
                 cache_images_in_memory=False,
                 ):
        """
        @param max_whole_image_size: the max size of the whole image
        @param use_dummy_images: use dummy images
        @param cache_images_in_memory: cache all images
        """
        super(COCOFormatDataset, self).__init__(root)
        self.max_whole_image_size = max_whole_image_size

        # don't load the image instead only return some empty black image
        self.use_dummy_images = use_dummy_images
        self.cache_images_in_memory = cache_images_in_memory

        self._load_dataset(annFile)

    def _load_dataset(self, annotation_file):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.shapes = []
        self.imgToAnns = dict()
        self.idxToImg = []
        self.n_c = 0
        self.max_objs_in_image = 0

        if annotation_file is not None:
            print('# Loading annotations into memory and parse...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self._create_index()
            print('# Done (t={:0.2f}s)'.format(time.time() - tic))

    def _shrink_image_to_max_size(self, img):
        if self.max_whole_image_size is None:
            img['whole_shrink_scale'] = 1.0
            return img

        width, height = img['width'], img['height']

        if height > width:
            scale = self.max_whole_image_size / height
            resized_height = self.max_whole_image_size
            resized_width = int(width * scale)
        else:
            scale = self.max_whole_image_size / width
            resized_height = int(height * scale)
            resized_width = self.max_whole_image_size

        if scale > 1.0:
            # we only shrink the larger images
            img['whole_shrink_scale'] = 1.0
            return img

        img['whole_shrink_scale'] = scale
        img['width'] = resized_width
        img['height'] = resized_height

        return img

    def _parse_categories(self):
        if isinstance(self.dataset['categories'], list):
            if isinstance(self.dataset['categories'][0], dict) and 'id' in self.dataset['categories'][0].keys():
                return {category['id']: category for category in self.dataset['categories']}
            else:
                return {_id: category for _id, category in enumerate(self.dataset['categories'])}

        if isinstance(self.dataset['categories'], dict):
            return self.dataset['categories']

    def _create_index(self):
        # create index
        print('# Creating index...')
        imgs = {}
        shapes = {}
        # FIXME there is a fixed number of 1000 splits per image!

        for img in self.dataset['images']:
            image_id = int(img['id'])

            img = self._shrink_image_to_max_size(img)
            imgs[image_id] = img.copy()
            imgs[image_id]['width'] = img['width']
            imgs[image_id]['height'] = img['height']

            self.imgToAnns[image_id] = []
            shapes[image_id] = ([imgs[image_id]['width'], imgs[image_id]['height']])

        self.ann_id = 0

        for ann in self.dataset['annotations']:
            image_id = int(ann['image_id'])
            whole_img_shrink_scale = imgs[image_id]['whole_shrink_scale']

            # search the correct img part
            bbox_x1 = ann['bbox'][0] * whole_img_shrink_scale
            bbox_y1 = ann['bbox'][1] * whole_img_shrink_scale
            bbox_width = ann['bbox'][2] * whole_img_shrink_scale
            bbox_height = ann['bbox'][3] * whole_img_shrink_scale

            self._add_annotation(bbox_x1, bbox_y1, bbox_width, bbox_height,
                                 ann['category_id'],
                                 image_id)

        self.n_c = len(self.dataset['categories'])

        self.cats = self._parse_categories()

        print('# Index created!')

        # create class members
        self.imgs = imgs
        self.shapes = np.array(shapes)
        self.max_objs_in_image = max([len(i) for i in self.imgToAnns.values()])
        self.idxToImg = list(self.imgs.keys())

        if self.cache_images_in_memory:
            self.cached_images = {}
            self._preload_images()

    def categories(self):
        return self.cats

    def _preload_images(self):
        for img_meta in tqdm.tqdm(self.imgs.values(), desc="Preload images"):
            filename = img_meta['file_name']
            if filename not in self.cached_images.keys():
                img = cv2.imread(os.path.join(self.root, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.cached_images[filename] = img

    def _add_annotation(self, x, y, width, height, label, img_id):
        # ignore too small bboxes
        if width <= 1 or height <= 1:
            return

        self.anns[self.ann_id] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'class_id': label,
        }
        self.imgToAnns[img_id].append(self.anns[self.ann_id])
        self.ann_id += 1

    def _get_annotations(self, index):
        image_id = self.idxToImg[index]
        # get ground truth annotations
        annotations = np.zeros((0, 5))

        for idx, a in enumerate(self.imgToAnns[image_id]):

            # some annotations have basically no width / height, skip them
            if a['width'] < 1 or a['height'] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [a['x'], a['y'], (a['x'] + a['width']), (a['y'] + a['height'])]
            annotation[0, 4] = a['class_id']
            annotations = np.append(annotations, annotation, axis=0)

        return torch.from_numpy(annotations)

    def load_image(self, index):
        image_id = self.idxToImg[index]
        img_meta = self.imgs[image_id]
        path = img_meta['file_name']

        if self.use_dummy_images:
            img = np.zeros((self.imgs[image_id]['width'], self.imgs[image_id]['height'], 3))
        else:
            if self.cache_images_in_memory:
                if path in self.cached_images.keys():
                    img = self.cached_images[path].copy()
                else:
                    img = cv2.imread(os.path.join(self.root, path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.cached_images[path] = img.copy()
            else:
                img = cv2.imread(os.path.join(self.root, path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            shrink_scale = img_meta['whole_shrink_scale']
            if shrink_scale != 1.0:
                img = cv2.resize(img, (img_meta['width'], img_meta['height']))

        if img is None:
            raise Exception("Could not load img: %s" % path)

        return torch.from_numpy(img)

    def get_image(self, index):
        image_id = self.idxToImg[index]
        annnotations = self._get_annotations(index)
        img = self.load_image(index)

        return DatasetElement(image_id, img, annnotations)

    def get_image_info(self, index):
        annnotations = self._get_annotations(index)
        image_id = self.idxToImg[index]
        image_meta = self.imgs[image_id]

        return DatasetElementInfo(image_id, image_meta['width'], image_meta['height'], annnotations)

    def __getitem__(self, index):
        return self.get_image(index)

    def __len__(self):
        return len(self.idxToImg)

    def num_classes(self):
        return self.n_c


def add_bbox_xyxy(image, left, top, right, bottom):
    if type(image) is not np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    image = cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

    return image


if __name__ == '__main__':
    data_path = 'example/'
    output_path = 'output/'

    alpha = 512
    beta = 0.25
    gamma = 0.5

    remove_empty_tiles = True
    add_full_frame = True

    print(f"#####")
    print(f"Tile size: (alpha): {alpha}")
    print(f"Tile overlap (beta): {beta}")
    print(f"Downscale factor for full frame (gamma): {gamma}")

    print(f"Remove empty tiles: {remove_empty_tiles}")
    print(f"Add full frame: {add_full_frame}")
    print(f"#####")

    os.makedirs(output_path, exist_ok=True)

    img_folder = os.path.join(data_path, "images")
    ann_file = os.path.join(data_path, "annotations", 'instances_all.json')

    dataset = COCOFormatDataset(img_folder, ann_file,
                                max_whole_image_size=None)
    dataset = CroWTiledDataset(dataset,
                               remove_empty_tiles=remove_empty_tiles,
                               add_full_frame=add_full_frame,
                               tile_size=alpha,
                               tile_overlapping=beta,
                               down_scale_factor=gamma)

    import matplotlib.pyplot as plt
    import tqdm

    np.random.seed(42)

    for i in tqdm.tqdm(np.random.randint(0, len(dataset), size=20), desc=f"Create example images in {os.path.abspath(output_path)}"):
        sample = dataset[i]
        image, annotations = sample.image, sample.annotation

        if not isinstance(image, np.ndarray):
            image = image.numpy()

        for bbox in annotations:
            image = add_bbox_xyxy(image, bbox[0], bbox[1], bbox[2], bbox[3])

        plt.imsave(os.path.join('output', f'image_{i}.png'), image)
