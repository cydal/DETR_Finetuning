from util.plot_utils import plot_logs
from pathlib import Path
from PIL import Image
from utils import *


if __name__ == "__main__":

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
                        num_classes=num_classes)

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
        run_worflow(im, model)


 