from utils import *



if __name__ == "__main__":


    load_pretrained_weights()
    coco_convert('/content/VIA2COCO/')

    copy_coco()


    plot_sample_images()