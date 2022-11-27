import glob
import pickle
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
from dataclasses import dataclass
from os.path import normpath
from typing import Union, Tuple, List, Any

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from config_ver1 import IMG_SIZE
from torch.utils.data import DataLoader
from test_network import load_model, get_results, TEMPLATE
from torch.utils.data import Dataset

color_map = {
    "LongCoat": (0, 145, 65),
    "ShirtNoCoat": (145, 65, 0),
    "TShirtNoCoat": (145, 0, 65),
    "Pants": (65, 0, 65),
    "ShortPants": (0, 65, 65)
}
numeric_map = {
    "Pants": 1,
    "ShortPants": 2,
    "ShirtNoCoat": 3,
    "TShiftNoCoat": 4,
    "LongCoat": 5
}


@dataclass
class Folder:
    path: str
    extensions: Union[Tuple, List]
    load_func: Any
    ignores: Union[Tuple, List] = tuple()
    sort_key: Any = None


class MirroredDataset(Dataset):
    def __init__(self, **kwargs):
        """
        kwargs should be Folder objects
        :param images_path:
        :param keypoints_path:
        """
        self.keys = kwargs.keys()
        for k, f in kwargs.items():
            setattr(self, k, f)

            files = []
            for ext in f.extensions:
                found_files = glob.glob(normpath(f.path) + "/**/*." + ext)
                files.extend(found_files)

            if not files:
                print("Warning: no files found for", f)
            files.sort(key=f.sort_key)

            setattr(self, k + "_files", files)

    def __getitem__(self, index):
        ret = dict()
        for k in self.keys:
            files_list = getattr(self, k + "_files")
            file_path = files_list[index]
            f_obj: Folder = getattr(self, k)
            loaded = f_obj.load_func(file_path)
            ret[k + "_path"] = file_path
            ret[k] = loaded

        return ret

    def __len__(self):
        lengths = [len(getattr(self, k + "_files")) for k in self.keys]
        return min(lengths)


def load_pgn_cloth_for_mgn(file_path):
    img = Image.open(file_path)
    img = make_square(img)  # pad to square
    img = img.resize((IMG_SIZE, IMG_SIZE))  # boost size to MGN required
    return np.array(img)


def vertex_label_from_cloth(file_path):
    img = Image.open(file_path)
    arr = np.array(img)
    found_items = []

    # searches the img for any occurences of the specified pixel value.
    # if found, remember that we found that garment item
    for key, color in color_map.items():
        indices = np.where(np.all(arr == color, axis=-1))[0]
        if len(indices) > 0:
            found_items.append(key)

    # from https://github.com/bharat-b7/MultiGarmentNetwork/issues/16#issuecomment-608986126
    vertexlabel = np.zeros((1, 27554, 1))
    for garment_name in found_items:
        vertexlabel[np.where(TEMPLATE[garment_name][1])[0]] = numeric_map[garment_name]

    return vertexlabel


def make_square(im: Image, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def load_openpose_from_file(resolution=(1, 1), person=0):
    """
    :param resolution: default=(1,1) means no normalization
    :param person:
    :return:
    """

    def inner(file):
        with open(file) as f:
            people = json.load(f)['people']
            if len(people) == 0:
                return np.zeros(
                    (25, 3))  # return an array of 0s; this is because 0 confidence
            data = people[person]
            try:
                kp = data["pose_keypoints_2d"]
            except KeyError:
                try:
                    kp = data["pose_keypoints"]
                except KeyError as e:
                    print("Wtf, something wrong is going on here")
                    raise e
            pose = np.array(kp).reshape(-1, 3)
            pose[:, 2] /= np.expand_dims(np.mean(pose[:, 2][pose[:, 2] > 0.1]), -1)
            pose = pose * np.array(
                [2. / resolution[1], -2. / resolution[0], 1.]) + np.array(
                [-1., 1., 0.])
            pose[:, 0] *= 1. * resolution[1] / resolution[0]

            return pose

    return inner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cloths", required=True,
                        help="path to cloth segmentations (RGB)")
    parser.add_argument("--keypoints", required=True, help="path to keypoints")
    parser.add_argument("--out", default="mgn_out", help="path to keypoints")
    parser.add_argument("--group_by_folder", action="store_true",
                        help="If passed, will treat the images in the lowest level "
                             "folder as different angles of the same subject+pose. "
                             "All images will be used for 3D extrapolation")

    args = parser.parse_args()

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=conf)

    m = load_model("saved_model/")

    ds = MirroredDataset(
        image_0=Folder(
            path=args.cloths,
            extensions=("png", "jpg"),
            load_func=load_pgn_cloth_for_mgn
        ),
        vertexlabel=Folder(
            path=args.cloths,
            extensions=("png", "jpg"),
            load_func=vertex_label_from_cloth
        ),
        J_2d_0=Folder(
            path=args.keypoints,
            extensions=("json",),
            load_func=load_openpose_from_file(resolution=(192, 256))
        ),
    )
    loader = DataLoader(ds)

    for batch in tqdm(loader, total=len(loader)):
        batch = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in
                 batch.items()}
        # print({k: v.shape if isinstance(v, np.ndarray) else v for k, v in batch.items()})
        paths_batch = batch["image_0_path"]
        top = len(normpath(args.cloths))
        if not np.any(batch["J_2d_0"]):
            tqdm.write("Skipping, no joints " + paths_batch[0])
        elif batch["J_2d_0"].shape != (1, 25, 3):
            tqdm.write(
                "Skipping, joints wrong shape " + str(batch["J_2d_0"].shape) + " " +
                paths_batch[0])
        else:
            tqdm.write("Running on " + paths_batch[0])

            pred = get_results(m, batch)
            # iterate over batch
            for i, instance in enumerate(pred):
                sample_id = os.path.splitext(paths_batch[i][top + 1:])[0]
                out_path = os.path.join(normpath(args.out), sample_id + ".pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(pred, f)
