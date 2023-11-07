import os
import json
import os
import cv2
import torch
from torchvision.ops import box_convert
import shutil


class LabelHandler:
    def __init__(self, data=None, td=None):
        assert data is not None and td is not None, 'Data and tool_dict must be provided.'
        self.data = data
        self.tool_dict = td
        self.named_data = {}
        self.named_label = {}
        self.group_label()

    def group_label(self):
        for data in self.data['annotations']:
            image_id = data['image_id']
            if image_id not in self.named_label:
                self.named_label[image_id] = []
            self.named_label[image_id].append(data)

    @staticmethod
    def path_to_query(p: str) -> str:
        return os.path.basename(p)

    def get_img_id(self, p_img: str) -> list[int]:
        if p_img in self.named_data:
            return self.named_data[p_img]

        for img_info in self.data['images'][:]:
            target_p = os.path.basename(img_info['file_name'])
            self.named_data[target_p] = [img_info['id'], img_info['width'], img_info['height']]
            self.data['images'].remove(img_info)
            if target_p == p_img:
                return [img_info['id'], img_info['width'], img_info['height']]
        raise ValueError(f'Image {p_img} not found in the dataset.')

    def get_labels_by_id(self, img_id: int, img_w: int, img_h: int) -> list[list[int | float]]:
        ls = self.named_label[img_id]
        new_labels = []
        for label in ls:
            bbox = label['bbox']
            cls_id = self.tool_dict['id2idx'][label['category_id']]
            new_labels.append([cls_id, *bbox, img_w, img_h])
        return new_labels

    def get_label(self, p_img: str) -> list[list[int | float]]:
        q_str = self.path_to_query(p_img)
        img_id, img_w, img_h = self.get_img_id(q_str)
        return self.get_labels_by_id(img_id, img_w, img_h)


if __name__ == '__main__':
    data_dir = 'G:/dataset/ACDC'
    target_dir = 'G:/dataset/ACDC/YOLO'

    gt_path = os.path.join(data_dir, 'gt_detection', 'instancesonly_train_gt_detection.json')
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    tool_dict = {
        'id2idx': {},
        'idx2id': {},
        'idx2name': {},
        'result': {'name': {}},
    }
    for idx, info in enumerate(gt_data['categories']):
        tool_dict['id2idx'][info['id']] = idx
        tool_dict['idx2id'][idx] = info['id']
        tool_dict['idx2name'][idx] = info['name']
        tool_dict['result']['name'][idx] = info['name']
    label_handler = LabelHandler(gt_data, tool_dict)

    img_dir = os.path.join(data_dir, 'rgb_anon')
    for subdir in os.listdir(img_dir):
        # make target dir
        target_subdir = os.path.join(target_dir, subdir)
        t_img_dir = os.path.join(target_subdir, 'images')
        t_label_dir = os.path.join(target_subdir, 'labels')
        os.makedirs(t_img_dir, exist_ok=True)
        os.makedirs(t_label_dir, exist_ok=True)
        # clear target dir
        for d in [t_img_dir, t_label_dir]:
            for f in os.listdir(d):
                if os.path.isfile(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
                else:
                    shutil.rmtree(os.path.join(d, f))

        sub_path = os.path.join(img_dir, subdir)
        gt_path = sub_path.replace('rgb_anon', 'gt_detection')
        split_set_names = os.listdir(sub_path)
        for split_name in split_set_names:
            if 'ref' in split_name or 'test' in split_name:
                continue
            gt_full_path = os.path.join(gt_path, f'instancesonly_{subdir}_{split_name}_gt_detection.json')
            split_path = os.path.join(sub_path, split_name)
            vid_seq_names = os.listdir(split_path)
            for vid_name in vid_seq_names:
                vid_path = os.path.join(split_path, vid_name)
                img_names = os.listdir(vid_path)
                for img_name in img_names:
                    img_path = os.path.join(vid_path, img_name)
                    labels = label_handler.get_label(img_path)
                    labels_t = torch.tensor(labels)
                    img_h, img_w = labels_t[0, -2], labels_t[0, -1]
                    x = labels_t[:, 1] / img_w
                    y = labels_t[:, 2] / img_h
                    w = labels_t[:, 3] / img_w
                    h = labels_t[:, 4] / img_h
                    bbox = box_convert(torch.stack([x, y, w, h], dim=1), 'xywh', 'cxcywh')
                    labels = torch.cat([labels_t[:, 0:1], bbox], dim=1).tolist()

                    # copy image
                    target_img_path = os.path.join(t_img_dir, f'{img_name}')
                    shutil.copy(img_path, target_img_path)
                    # write label
                    target_label_path = os.path.join(t_label_dir, f'{img_name.replace(".png", ".txt")}')
                    with open(target_label_path, 'w', encoding='utf-8') as f:
                        for label in labels:
                            f.write(f'{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n')

