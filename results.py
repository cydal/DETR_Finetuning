from util.plot_utils import plot_logs
from pathlib import Path
from PIL import Image
from utils import *
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt



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


def filter_bboxes_from_outputs(im, outputs,
                               threshold=0.7):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  return probas_to_keep, bboxes_scaled


def run_worflow(my_image, my_model, transform):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)

  # propagate through the model
  outputs = my_model(img)
    
  probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(img, outputs,
                                                            threshold=0.9)

  plot_finetuned_results(my_image,
                          probas_to_keep, 
                          bboxes_scaled)




def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    log_directory = [Path('outputs/')]


    fields_of_interest = (
        'loss',
        'mAP',
        )

    fields_of_interest = (
        'loss_ce',
        'loss_bbox',
        'loss_giou',
        )

    fields_of_interest = (
        'class_error',
        'cardinality_error_unscaled',
        )

    plot_logs(log_directory,
            fields_of_interest)


    plot_logs(log_directory,
            fields_of_interest)

    plot_logs(log_directory,
            fields_of_interest)   


    model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=1)

    checkpoint = torch.load('outputs/checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                        strict=False)

    model.eval()



    img_names = ['/content/data/custom/val2017/3825919971_93fb1ec581_b.jpg', 
                '/content/data/custom/val2017/6810773040_3d81036d05_k.jpg'
                '/content/data/custom/val2017/4838031651_3e7b5ea5c7_b.jpg']

    for img in img_names:

        im = Image.open(img)
        run_worflow(im, model, transform)


 