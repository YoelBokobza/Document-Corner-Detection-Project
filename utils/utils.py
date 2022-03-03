import os
import numpy as np
from PIL import Image
from matplotlib.path import Path
import xml.etree.ElementTree as ET
import numbers
from torchvision.transforms import GaussianBlur, Compose
import torch
import random



def sort_predication(masks_pred):
    """
    The following function receives our corner estimation and sort it according to a predefined sequence
    [Top Left, Top Right, Bottom Right, Bottom Left]
    """
    sort_idx = np.argsort(masks_pred[:, 0])
    tl_idx = np.argmin(masks_pred[sort_idx][:2][:, 1])
    tr_idx = np.argmax(masks_pred[sort_idx][:2][:, 1])
    tl = masks_pred[sort_idx][:2][tl_idx]
    tr = masks_pred[sort_idx][:2][tr_idx]
    bl_idx = np.argmin(masks_pred[sort_idx][2:][:, 1])
    br_idx = np.argmax(masks_pred[sort_idx][2:][:, 1])
    bl = masks_pred[sort_idx][2:][bl_idx]
    br = masks_pred[sort_idx][2:][br_idx]
    return np.array([tl, tr, br, bl])

def gen_mask(coordinates, sizeX, sizeY, sort):
    '''
    The following function generates a binary mask image given 4 2-dimensional coordinates.
    '''


    nx, ny = sizeX, sizeY
    if sort:
        coordinates = sort_predication(coordinates)
        coordinates = list(map(tuple, coordinates))

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    coordinates.append((0., 0.))
    poly_verts = coordinates
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    y, x = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts, codes=codes)
    grid = path.contains_points(points)
    mask = grid.reshape((ny, nx)).T
    return mask

def sort_gt(gt):
    '''
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt:
    :return: sorted gt
    '''
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    return np.asarray((tl, tr, br, bl))



class _Dataset():
    '''
    Base class to reprenent a Dataset
    '''
    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []

class SmartDocDirectories(_Dataset):
    '''
    Class to include SmartDoc Dataset via full resolution images while sampling only 2 images from a single video
    '''

    def __init__(self, frames_per_video, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                if os.path.isdir(directory+"/"+folder+"/"+os.listdir(directory + "/" + folder)[0]):
                    for file in os.listdir(directory + "/" + folder):
                        images_dir = directory + "/" + folder + "/" + file
                        self.read_background(images_dir,frames_per_video, file=file)                
                else:
                    images_dir = directory + "/" + folder
                    self.read_background(images_dir, frames_per_video, file=folder)


        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        print("Ground Truth Shape: %s", str(self.labels.shape))
        print("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

    def read_background(self, images_dir, frames_per_video, file):

        if (os.path.isdir(images_dir)):
          list_gt = []
          tree = ET.parse(images_dir + "/" + file + ".gt")
          root = tree.getroot()
          for a in root.iter("frame"):
              list_gt.append(a)

          im_no = 0
          # List all available images in a list
          images_list = os.listdir(images_dir)
          # Remove from the list the Ground True
          images_list.remove(file + ".gt")
          
          if frames_per_video == 'all':
            ChosenImages = list(images_list)
          else:
            ChosenImages = list(np.random.choice(images_list, size=frames_per_video))
          for image in ChosenImages:
              im_no += 1
              # Now we have opened the file and GT. Write code to create multiple files and scale gt
              list_of_points = {}

              self.data.append(os.path.join(images_dir, image))
              imageIndex = int(float(image[0:-4])) - 1
              for point in list_gt[imageIndex].iter("point"):
                  myDict = point.attrib

                  list_of_points[myDict["name"]] = (int(float(myDict['x'])), int(float(myDict['y'])))

              ground_truth = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"], list_of_points["bl"]))
              ground_truth = sort_gt(ground_truth)
              self.labels.append(ground_truth)


def generate_full_resolution_partial_dataset(path,frames_per_video, size):
    '''
    This function generate a dataset from a given path of images and sample a predefined number of random frames from each video.
    Our dataset contain 3 main objects.
    - Raw RGB Images.
    - Ground Truth Binary Masks.
    - Ground Truth Corners Coordinates.
    It also calculates the first and second statisticals moments of the given dataset.
    '''
    dataset = SmartDocDirectories(directory=path, frames_per_video=frames_per_video)
    images = []
    labels = []
    masks = []
    X_Indices = [0, 2, 4, 6]
    Y_Indices = [1, 3, 5, 7]
    for img_path, label in dataset.myData:
        img = Image.open(img_path)
        old_size = img.size
        x_cords = label[X_Indices] * (size / old_size[0])
        y_cords = label[Y_Indices] * (size / old_size[1])
        img = img.resize((size, size))
        img = np.array(img)
        currLabel = [(i, j) for i, j in zip(x_cords, y_cords)]
        normLabel = [(float(x) /old_size[0], float(y)/old_size[1]) for x,y in zip(label[X_Indices],label[Y_Indices])]
        normLabel = list(sum(normLabel,()))
        mask = gen_mask(currLabel, sizeX=img.shape[0], sizeY=img.shape[1], sort=False)
        labels.append(normLabel)
        images.append(img.T)
        masks.append(mask.T)
    stats = {'mean': np.mean(np.array(images)), 'std': np.std(np.array(images))}

    return images, masks, labels, stats



class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.T.astype('uint8'))

        corners = np.array(list(map(lambda x: np.array((x[0] * self.size[0] / img.size[0], x[1] * self.size[1] / img.size[1])), corners)))

        resized_img = img.resize(self.size)
        resized_mask = mask.resize(self.size)

        return np.array(resized_img).T, np.array(resized_mask).T, corners


class Normalize(object):
    def __init__(self, *args):
        self.mean, self.std = tuple(*args)

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = (img - self.mean) / self.std

        return img, mask, corners


class ourGaussianBlur(object):
    def __init__(self, size):
        self.filter = GaussianBlur(size)

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        img = np.array(self.filter(img)).T

        return img, mask, corners

class RandomCropNearCorner(object):
    '''
    This class conatins our transformation for random crop around a random document's corner.
    We shall notice that this transformation is used only for the training paradigm
    '''
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.T.astype('uint8'))
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img
        corner = random.randint(0, 3)
        x_center = corners[(corner*2)] * w
        y_center = corners[(corner*2)+1] * h
        randTw = random.randint(0, tw)
        randTh = random.randint(0, th)
        x1 = max(x_center - randTw, 0)
        y1 = max(y_center - randTh, 0)
        x2 = min(x1 + tw, w)
        y2 = min(y1 + th, h)
        if x2 == w:
            x1 = w - tw
        if y2 == h:
            y1 = h - th
        cropped_img = np.array(img.crop((x1, y1, x2, y2))).T
        cropped_mask = np.array(mask.crop((x1, y1, x2, y2))).T
        new_cor = [(corners[(corner*2)] * w - x1) / tw, (corners[(corner*2)+1] * h - y1) / th]

        return cropped_img, cropped_mask, new_cor






class DocumentDatasetMaskSegmentation(object):
    '''
    This is our dataset implementation for our custom case of 3 different objects for each image (RGB Image, Mask, Coordinates)
    '''
    def __init__(self, TransformsMapping, Path: str, frames_per_video, Transforms: dict, Size : int):
        self.images, self.masks, self.labels, self.stats = generate_full_resolution_partial_dataset(Path, frames_per_video, size=Size)
        self.withTransforms = len(Transforms) != 0
        if Transforms['Normalize'] is None:
            Transforms['Normalize'] = (self.stats['mean'], self.stats['std'])
        transform_list = [TransformsMapping[transform](value) for transform, value in Transforms.items()]
        self.transforms = Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.withTransforms:
            image, mask, label = self.transforms((self.images[idx], self.masks[idx], self.labels[idx]))
        else:
            image, mask, label = self.images[idx], self.masks[idx], self.labels[idx]
        return {
            'image': torch.as_tensor(image).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(label.copy()).float().contiguous()
        }
    
