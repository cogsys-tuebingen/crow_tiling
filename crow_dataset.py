import torchvision
import torch
import os
import numpy as np
import cv2
import enum
import tqdm


class CutHandling(enum.Enum):
    REMOVE_SPLITS = "remove_splits",
    IGNORE_CUT_OBJECTS = "ignore_cut_objects"


class DatasetElementInfo:
    def __init__(self, image_id: int, width: int, height: int, annotation: torch.Tensor):
        self.image_id = image_id
        self.width = width
        self.height = height
        self.annotation = annotation


class DatasetElement:
    def __init__(self, image_id: int, image: torch.Tensor, annotation: torch.Tensor):
        self.image = image
        self.annotation = annotation
        self.image_id = image_id


class BaseDataset:
    def __len__(self) -> int:
        raise NotImplementedError()

    def num_classes(self) -> int:
        """
            returns a number of classes
        """
        raise NotImplementedError()

    def __getitem__(self, index) -> DatasetElement:
        raise NotImplementedError()

    def get_image_info(self, index) -> DatasetElementInfo:
        raise NotImplementedError()

    def categories(self):
        raise NotImplementedError()

    def load_image(self, index) -> torch.Tensor:
        raise NotImplementedError()


def generate_tile_positions(w, h, sq, overlap=0.25):
    """
    create the tile positions for a image with the size w x h and the tile size
    of sq x sq

    @return a list of square splits (x1, y1, w, h)
    """

    if overlap == 0:
        xs = [i * sq for i in range(int(w / sq) + 1)]
        ys = [i * sq for i in range(int(h / sq) + 1)]

    else:
        # add the first and last box positions
        xs = [0, w - sq]
        ys = [0, h - sq]

        # add the overlapping boxes
        distance_between_x_splits = sq * (1 - overlap)
        distance_between_y_splits = sq * (1 - overlap)

        x_inter_split_num = int((xs[-1] - xs[0]) / distance_between_x_splits)
        if overlap == 0:
            x_inter_split_num -= 1
            x_inter_split_num = max(0, x_inter_split_num)

        y_inter_split_num = int((ys[-1] - ys[0]) / distance_between_y_splits)
        if overlap == 0:
            y_inter_split_num -= 1
            y_inter_split_num = max(0, y_inter_split_num)

        xs += [int((i * distance_between_x_splits + (sq / 2))) for i in range(x_inter_split_num)]
        ys += [int((i * distance_between_y_splits + (sq / 2))) for i in range(y_inter_split_num)]

        # move the last box position to the end
        xs.append(xs.pop(1))
        ys.append(ys.pop(1))

    xs, ys = np.array(xs), np.array(ys)
    p = np.stack(np.meshgrid(xs, ys)).T.astype(int)
    p_filled = np.zeros((p.shape[0], p.shape[1], p.shape[2] + 2), dtype=int)
    p_filled[:, :, :-2] = p
    p_filled[:, :, -2:] = sq
    p = p_filled

    return p


class CroWTiledDataset(BaseDataset, torchvision.datasets.VisionDataset):
    def __init__(self, org_dataset: BaseDataset,
                 tile_size=512,
                 tile_overlapping=0.25,
                 down_scale_factor=1,
                 add_full_frame: bool = True,
                 remove_empty_tiles=True,
                 handle_cut_objects: CutHandling = False,
                 transform=None):
        """
    @param split_size(alpha): tile size
    @param tile_overlapping(beta): min overlap between tiles
    @param down_scale_factor(gamma): the down_scale_factore for the full frame (0..1)

    @param remove_empty_tiles: whether tiles with no object should be removed?
    @param handle_cut_objects: how to handle objects which are cut (only used for ablation study)
    @param add_full_frame: whether the whole image should also be a element of the dataset.

    @param transform: torch transformation for the samples
    """
        super(CroWTiledDataset, self).__init__("", transform=transform)

        self.org_dataset = org_dataset
        self.remove_empty_imgs = remove_empty_tiles
        self.handle_cut_objects = handle_cut_objects
        self.add_whole_image = add_full_frame
        self.split_size = tile_size
        self.tile_overlapping = tile_overlapping
        self.down_scale_factor = down_scale_factor

        self._createIndex()

    def __getitem__(self, index) -> DatasetElement:
        return self._get_img(index)

    def __len__(self):
        return len(self.idxToImg)

    def num_classes(self):
        return self.org_dataset.num_classes()

    def _createIndex(self):
        # create index
        imgs = {}
        shapes = {}
        self.img_parts = {}
        self.ann_id = 0
        self.anns = {}
        self.imgToAnns = {}
        self.idxToPart = {}

        # FIXME: find a smart way to find the max
        max_splits = 0
        for i in range(len(self.org_dataset)):
            image_info = self.org_dataset.get_image_info(i)

            squares = generate_tile_positions(int(image_info.width),
                                              int(image_info.height),
                                              int(self.split_size),
                                              overlap=self.tile_overlapping)
            splits = squares.shape[0] * squares.shape[1]
            max_splits = max(splits, max_splits)

        print(f"# There are max {max_splits} per image")
        # Use the max number of splits
        self.splits_per_img = max_splits * 10

        for i in tqdm.tqdm(range(len(self.org_dataset)), desc="Create tiles.."):
            image_info = self.org_dataset.get_image_info(i)
            original_image_index = i
            original_image_id = image_info.image_id

            splitted_start_id = int(self.splits_per_img * original_image_id)
            self.img_parts[splitted_start_id] = []

            squares = generate_tile_positions(int(image_info.width),
                                              int(image_info.height),
                                              int(self.split_size),
                                              overlap=self.tile_overlapping)

            for line_id, line in enumerate(squares):
                for part_id, part in enumerate(line):
                    img_id = int(splitted_start_id + part_id + (line_id * len(line)))
                    imgs[img_id] = {}
                    imgs[img_id]['width'] = part[2]
                    imgs[img_id]['height'] = part[3]
                    imgs[img_id]['part_info'] = {
                        'x_1': part[0],
                        'y_1': part[1],
                        'x_2': part[0] + part[2],
                        'y_2': part[1] + part[3],
                        'img_id_x': line_id,
                        'img_id_y': part_id,
                        'img_id': img_id,
                        'img_width': image_info.width,
                        'img_height': image_info.height,
                        'original_image_index': original_image_index,
                        'downscale_factor': 1
                    }
                    self.imgToAnns[img_id] = []
                    shapes[img_id] = ([imgs[img_id]['width'], imgs[img_id]['height']])
                    self.idxToPart[img_id] = (line_id, part_id)

                    self.img_parts[int(splitted_start_id)].append(
                        imgs[img_id]['part_info']
                    )

            if self.add_whole_image:
                # also add the whole img like in the power of tiling
                img_id = splitted_start_id + (self.splits_per_img - 1)
                imgs[img_id] = {}
                imgs[img_id]['width'] = int(image_info.width * self.down_scale_factor)
                imgs[img_id]['height'] = int(image_info.height * self.down_scale_factor)
                imgs[img_id]['part_info'] = {
                    'x_1': 0,
                    'y_1': 0,
                    'x_2': int(image_info.width * self.down_scale_factor),
                    'y_2': int(image_info.height * self.down_scale_factor),
                    'img_id_x': -1,
                    'img_id_y': -1,
                    'img_id': img_id,
                    'img_width': int(image_info.width * self.down_scale_factor),
                    'img_height': int(image_info.height * self.down_scale_factor),
                    'original_image_index': original_image_index,
                    'downscale_factor': self.down_scale_factor
                }
                self.imgToAnns[img_id] = []
                shapes[img_id] = ([imgs[img_id]['width'], imgs[img_id]['height']])
                self.idxToPart[img_id] = (-1, -1)

                self.img_parts[int(splitted_start_id)].append(
                    imgs[img_id]['part_info']
                )

            annotations = image_info.annotation
            base_img_id = int(self.splits_per_img * original_image_id)
            for ann in annotations:
                # search the correct img part
                bbox_x1 = ann[0]
                bbox_y1 = ann[1]
                bbox_x2 = ann[2]
                bbox_y2 = ann[3]
                class_id = ann[4]

                bbox_width = bbox_x2 - bbox_x1
                bbox_height = bbox_y2 - bbox_y1

                fitting_parts = []
                for img_part in self.img_parts[base_img_id]:
                    part_x1, part_y1, part_x2, part_y2 = img_part['x_1'], img_part['y_1'], img_part['x_2'], img_part[
                        'y_2']

                    _bbox_x1 = bbox_x1
                    _bbox_x2 = bbox_x2
                    _bbox_y1 = bbox_y1
                    _bbox_y2 = bbox_y2

                    if 'downscale_factor' in img_part and img_part['downscale_factor'] != 1:
                        _bbox_x1 = _bbox_x1 * img_part['downscale_factor']
                        _bbox_x2 = _bbox_x2 * img_part['downscale_factor']
                        _bbox_y1 = _bbox_y1 * img_part['downscale_factor']
                        _bbox_y2 = _bbox_y2 * img_part['downscale_factor']

                    _bbox_width = _bbox_x2 - _bbox_x1
                    _bbox_height = _bbox_y2 - _bbox_y1

                    if _bbox_x1 > part_x2 or _bbox_x2 < part_x1:
                        continue

                    if _bbox_y1 > part_y2 or _bbox_y2 < part_y1:
                        continue

                    fitting_parts.append(img_part)

                # now add the annotation for each match
                for match in fitting_parts:
                    part_x1, part_y1, part_x2, part_y2 = match['x_1'], match['y_1'], match['x_2'], match['y_2']

                    _bbox_x1 = bbox_x1
                    _bbox_x2 = bbox_x2
                    _bbox_y1 = bbox_y1
                    _bbox_y2 = bbox_y2

                    if 'downscale_factor' in match and match['downscale_factor'] != 1:
                        _bbox_x1 = _bbox_x1 * match['downscale_factor']
                        _bbox_x2 = _bbox_x2 * match['downscale_factor']
                        _bbox_y1 = _bbox_y1 * match['downscale_factor']
                        _bbox_y2 = _bbox_y2 * match['downscale_factor']

                    _bbox_width = _bbox_x2 - _bbox_x1
                    _bbox_height = _bbox_y2 - _bbox_y1

                    _x_in_part = max(_bbox_x1 - part_x1, 0)
                    _y_in_part = max(_bbox_y1 - part_y1, 0)

                    _x2 = _bbox_x1 + _bbox_width
                    _y2 = _bbox_y1 + _bbox_height

                    _x2_in_part = min(_x2 - part_x1, part_x2 - part_x1)
                    _y2_in_part = min(_y2 - part_y1, part_y2 - part_y1)

                    actual_bbox_width = _x2_in_part - _x_in_part
                    actual_bbox_height = _y2_in_part - _y_in_part

                    is_cut = (int(actual_bbox_width) < int(bbox_width)) or (int(actual_bbox_height) < int(bbox_height))

                    self._add_annotation(_x_in_part, _y_in_part, _x2_in_part - _x_in_part, _y2_in_part - _y_in_part,
                                         class_id,
                                         match['img_id'],
                                         is_cut=is_cut)

        self.empty_img_ids = [i for i in list(imgs.keys()) if len(self.imgToAnns[i]) == 0]
        print(f"Empty images[{len(self.empty_img_ids)}]: {self.empty_img_ids}")

        if self.remove_empty_imgs:
            # throw all images without annotation away
            for img_idx in self.empty_img_ids:
                imgs.pop(img_idx)
                self.imgToAnns.pop(img_idx)

        if self.handle_cut_objects == CutHandling.REMOVE_SPLITS:
            for img_idx in list(imgs.keys()):
                has_cut_annot = np.any([a['is_cut'] for a in self.imgToAnns[img_idx]])
                if has_cut_annot:
                    imgs.pop(img_idx)

        if self.handle_cut_objects == CutHandling.IGNORE_CUT_OBJECTS:
            for img_idx in list(imgs.keys()):
                has_uncut_annot = np.any([not a['is_cut'] for a in self.imgToAnns[img_idx]])
                if not has_uncut_annot:
                    imgs.pop(img_idx)

        print('# Index created!')

        # create class members
        self.imgs = imgs
        self.shapes = np.array(shapes)
        self.max_objs_in_image = max([len(i) for i in self.imgToAnns.values()])
        self.idxToImg = list(self.imgs.keys())

    def _add_annotation(self, x, y, width, height, label, img_id, is_cut):
        # ignore too small bboxes
        if width <= 1 or height <= 1:
            return

        self.anns[self.ann_id] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'class_id': label,
            'is_cut': is_cut
        }
        self.imgToAnns[img_id].append(self.anns[self.ann_id])
        self.ann_id += 1

    def categories(self):
        return self.org_dataset.categories()

    def _get_annotations(self, index):
        image_id = self.idxToImg[index]
        # get ground truth annotations
        annotations = np.zeros((0, 6))

        for idx, a in enumerate(self.imgToAnns[image_id]):

            # some annotations have basically no width / height, skip them
            if a['width'] < 1 or a['height'] < 1:
                continue

            annotation = np.zeros((1, 6))
            annotation[0, :4] = [a['x'], a['y'], (a['x'] + a['width']), (a['y'] + a['height'])]
            annotation[0, 4] = a['class_id']
            annotation[0, 5] = a['is_cut']
            annotations = np.append(annotations, annotation, axis=0)

        return torch.from_numpy(annotations)

    def _get_img(self, index):
        image_id = self.idxToImg[index]
        annotations = self._get_annotations(index)

        # get correct part of the image
        img_meta = self.imgs[image_id]
        image = self.org_dataset.load_image(img_meta['part_info']['original_image_index'])

        # downscale with gamma
        if img_meta['part_info']['downscale_factor'] != 1:
            image = torch.from_numpy(cv2.resize(image.numpy(),
                                           (img_meta['part_info']['img_width'],
                                            img_meta['part_info']['img_height'])))

        _part_x, _part_y = self.idxToPart[image_id]
        image = image[img_meta['part_info']['y_1']:img_meta['part_info']['y_2'],
                 img_meta['part_info']['x_1']:img_meta['part_info']['x_2'], :]

        if self.handle_cut_objects == CutHandling.IGNORE_CUT_OBJECTS:
            # set the cut objects to black and remove their labels
            is_cut_mask = annotations[:, 5] > 0
            for (_x, _y, _x2, _y2) in annotations[is_cut_mask, :4]:
                image[int(_y): int(_y2), int(_x):int(_x2), :] = 0
            annotations = annotations[np.logical_not(is_cut_mask)]

        if image.shape[0] < self.split_size and image.shape[1] < self.split_size:
            # if the original image is smaller then the split size, there can be problems
            # it is filled up with black
            # for example for the vis drone dataset
            extended_image = np.zeros((self.split_size, self.split_size, 3), dtype=np.uint8)
            extended_image[:image.shape[0], :image.shape[1]] = image
            image = extended_image

        # remove the is_cut column
        annotations = annotations[:, :5]

        sample = DatasetElement(image_id, image, annotations)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image_info(self, index):
        annotations = self._get_annotations(index)
        image_id = self.idxToImg[index]
        image_meta = self.imgs[image_id]

        return DatasetElementInfo(image_id, image_meta['width'], image_meta['height'], annotations)
