import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import torch, torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from subprocess import call
import os


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs,
                               threshold=0.7):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  return probas_to_keep, bboxes_scaled



def plot_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()



def load_pretrained_weights():
    # Get pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
                url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                map_location='cpu',
                check_hash=True)

    # Remove class weights
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # Save
    torch.save(checkpoint,
            '/content/detr/detr-r50_no-class-head.pth')


def coco_convert(data_path):


    #'/content/VIA2COCO/'

    for keyword in ['train', 'val']:

        input_dir = data_path + 'balloon/' + keyword + '/'
        input_json = input_dir + 'via_region_data.json'
        categories = ['balloon']
        super_categories = ['N/A']
        output_json = input_dir + 'custom_' + keyword + '.json'

        print('Converting {} from VIA format to COCO format'.format(input_json))

        coco_dict = via2coco.convert(
            imgdir=input_dir,
            annpath=input_json,
            categories=categories,
            super_categories=super_categories,
            output_file_name=output_json,
            first_class_index=first_class_index,
        )

def plot_sample_images():

    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    dataDir='/content/data/custom/'
    dataType='train2017'
    annFile='{}annotations/custom_train.json'.format(dataDir)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    nms=[cat['name'] for cat in cats]
    print('Categories: {}'.format(nms))

    nms = set([cat['supercategory'] for cat in cats])
    print('Super-categories: {}'.format(nms))

    # load and display image
    catIds = coco.getCatIds(catNms=['balloon'])
    imgIds = coco.getImgIds(catIds=catIds)

    img_id = imgIds[np.random.randint(0,len(imgIds))]
    print('Image n°{}'.format(img_id))

    img = coco.loadImgs(img_id)[0]

    img_name = '%s/%s/%s'%(dataDir, dataType, img['file_name'])
    print('Image name: {}'.format(img_name))

    I = io.imread(img_name)

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)

    plt.imshow(I)
    coco.showAnns(anns, draw_bbox=True)


def call_command(arg1, arg2=None):
  if arg2 is not None:
    call(['mv', '-a', arg1, arg2])
  else:
    call(['mkdir', '-p', arg1])

def copy_coco():
    call_command('/content/data/custom/annotations/', arg2=None)
    call_command('/content/data/custom/train2017/', arg2=None)
    call_command('/content/data/custom/val2017/', arg2=None)
    

    call_command('/content/VIA2COCO/balloon/train/custom_train.json', 
                 '/content/data/custom/annotations/custom_train.json')

    call_command('/content/VIA2COCO/balloon/val/custom_val.json', 
                 '/content/data/custom/annotations/custom_val.json')
    
    call_command('/content/VIA2COCO/balloon/train/*.jpg', 
                 '/content/data/custom/train2017/')

    call_command('/content/VIA2COCO/balloon/val/*.jpg', 
                 '/content/data/custom/val2017/)
    