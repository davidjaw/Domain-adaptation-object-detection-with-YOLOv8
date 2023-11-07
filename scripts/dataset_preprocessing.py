import json
import os
import cv2
import torch
from torchvision.ops import box_convert
import shutil
import yaml


class LabelHandler:
    def __init__(self, train_d, val_d):
        assert train_d is not None and val_d is not None, 'data, train_d, val_d must be provided.'
        tool_dict = {
            'id2idx': {},
            'idx2id': {},
            'idx2name': {},
            'result': {'name': {}, 'path': target_dir},
        }
        for idx, info in enumerate(train_d['categories']):
            tool_dict['id2idx'][info['id']] = idx
            tool_dict['idx2id'][idx] = info['id']
            tool_dict['idx2name'][idx] = info['name']
            tool_dict['result']['name'][idx] = info['name']
        self.tool_dict = tool_dict
        self.d = {'train': train_d, 'val': val_d}
        self.named_table = {
            'train': {
                'images': {},
                'labels': {},
            },
            'val': {
                'images': {},
                'labels': {},
            },
        }
        self.group_label()

    def id2name(self, _id: int) -> str:
        return self.tool_dict['idx2name'][_id]

    def group_label(self):
        for t in self.d.keys():
            data = self.d[t]
            for img_info in data['images']:
                img_id = img_info['id']
                i_name = os.path.basename(img_info['file_name'])
                self.named_table[t]['images'][i_name] = [img_id, img_info['width'], img_info['height']]
            for label_info in data['annotations']:
                img_id = label_info['image_id']
                if img_id not in self.named_table[t]['labels']:
                    self.named_table[t]['labels'][img_id] = []
                self.named_table[t]['labels'][img_id].append(label_info)

    @staticmethod
    def path_to_query(p: str) -> str:
        return os.path.basename(p)

    def get_img_id(self, p_img: str, dict_key: str) -> list[int]:
        q_dict = self.named_table[dict_key]
        if p_img in q_dict['images']:
            return q_dict['images'][p_img]
        raise ValueError(f'Image {p_img} not found in the dataset.')

    def get_labels_by_id(self, img_id: int, img_w: int, img_h: int, k: str) -> list[list[int | float]]:
        q_data = self.named_table[k]
        if img_id not in q_data['labels']:
            return []
        ls = q_data['labels'][img_id]
        new_labels = []
        for label in ls:
            bbox = label['bbox']
            cls_id = self.tool_dict['id2idx'][label['category_id']]
            new_labels.append([cls_id, *bbox, img_w, img_h])
        return new_labels

    def get_label(self, p_img: str, type_: str) -> list[list[int | float]]:
        q_str = self.path_to_query(p_img)
        img_id, w, h = self.get_img_id(q_str, type_)
        label = self.get_labels_by_id(img_id, w, h, type_)
        return label


if __name__ == '__main__':
    data_dir = 'G:/dataset/ACDC'
    target_dir = 'G:/dataset/ACDC/YOLO'

    gt_path = os.path.join(data_dir, 'gt_detection', 'instancesonly_train_gt_detection.json')
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    val_gt_path = os.path.join(data_dir, 'gt_detection', 'instancesonly_val_gt_detection.json')
    with open(val_gt_path, 'r', encoding='utf-8') as f:
        val_gt_data = json.load(f)
    label_handler = LabelHandler(gt_data, val_gt_data)

    img_dir = os.path.join(data_dir, 'rgb_anon')
    for subdir in os.listdir(img_dir):
        # make target dir
        for t in ['train', 'val']:
            target_subdir = os.path.join(target_dir, t, subdir)
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
            split_path = os.path.join(sub_path, split_name)
            vid_seq_names = os.listdir(split_path)
            for vid_name in vid_seq_names:
                vid_path = os.path.join(split_path, vid_name)
                img_names = os.listdir(vid_path)
                for img_name in img_names:
                    img_path = os.path.join(vid_path, img_name)
                    labels = label_handler.get_label(img_path, split_name)

                    if len(labels) > 0:
                        # process labels to meet YOLO format
                        labels_t = torch.tensor(labels)
                        img_w, img_h = labels_t[0, -2], labels_t[0, -1]
                        x = labels_t[:, 1] / img_w
                        y = labels_t[:, 2] / img_h
                        w = labels_t[:, 3] / img_w
                        h = labels_t[:, 4] / img_h
                        bbox = box_convert(torch.stack([x, y, w, h], dim=1), 'xywh', 'cxcywh')
                        labels = torch.cat([labels_t[:, 0:1], bbox], dim=1).tolist()
                    else:
                        print(f'No label found for {img_path}, skipped.')
                        continue

                    # copy image to target dir
                    t_img_dir = os.path.join(target_dir, split_name, subdir, 'images')
                    t_label_dir = os.path.join(target_dir, split_name, subdir, 'labels')
                    target_img_path = os.path.join(t_img_dir, f'{img_name}')
                    shutil.copy(img_path, target_img_path)
                    # write label
                    target_label_path = os.path.join(t_label_dir, f'{img_name.replace(".png", ".txt")}')
                    with open(target_label_path, 'w', encoding='utf-8') as f:
                        for label in labels:
                            f.write(f'{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n')

        for t in ['train', 'val']:
            p = t_img_dir.replace(target_dir, '').replace('train', t).replace('val', t)
            while p.startswith('/') or p.startswith('\\'):
                p = p[1:]
            label_handler.tool_dict['result'][t] = p
        if os.getcwd().endswith('scripts'):
            os.chdir('..')
        target_yaml = os.path.join('data', f'ACDC-{subdir}.yaml')
        with open(target_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(label_handler.tool_dict['result'], f)


