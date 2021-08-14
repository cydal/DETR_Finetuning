# DETR_Finetuning
finetuning of DETR on Balloon dataset



### Convert Annotation from VIA to COCO format
> git clone https://github.com/woctezuma/VIA2COCO
cd VIA2COCO/
git checkout fixes

* Download finetune dataset into VIA2COCO folder.
* Copy prep.py and utils.py into VIA2COCO folder and run to initiate conversion & DETR pre-trained weight download.


### Prepare dataset

> python prep.py


### Finetune on dataset
> python main.py \
--dataset_file "custom" \
--coco_path "/content/data/custom/" \
--output_dir "outputs" \
--resume "detr-r50_no-class-head.pth" \
--num_classes 1 \
--epochs 100


### Visualize model performance
> python results.py

* Copy results.py to DETR folder

### DETR
DETR is an end to end object detection architecture for object detection using transformers. It is replaces a lot of the manual processes typically noticed in earlier detection architectures like NMS and anchor box generation with learnable methods. 

[![image.png](https://i.postimg.cc/prCQ1SyG/image.png)](https://postimg.cc/QKBWF0Q1)

### Encoder-Decoder architecture
DETR utilizes an encoder-decoder architecture where the encoder takes in the input image embeddings and this is passed through a self-attention mechanism. In the decoder, object queries are passed in as the input. The object queries interact with the transformed input embeddings that is the output of the encoder and ultimately get transformed into the required output. In the case of object detection,  bounding boxes as well as class predictions. 

### Bipartite loss, and why we need it
The output of DETR is a set of predictions the same size as the object queries. The predictions don't come in any particular order, and multiple predictions may refer to the same output. The Bipartite matching loss obtains the loss by matching a single prediction with the ground truth it matches the most. The loss then is that which results when every prediction is matched with the single ground truth that it most matches. 

### Object Queries
This is the learnable input to the decoder. The object queries interact with the output of the encoder and ultimately get transformed to the prediction output.


### Training Log

[![image.png](https://i.postimg.cc/kGd5rYKr/image.png)](https://postimg.cc/cgXWwmFT)


### Model Performance

[![image.png](https://i.postimg.cc/W1MCbNDx/image.png)](https://postimg.cc/jDSZMr4h)

[![image.png](https://i.postimg.cc/FRKBZ0PQ/image.png)](https://postimg.cc/Jy9x7HVT)

[![image.png](https://i.postimg.cc/d1SSSDXZ/image.png)](https://postimg.cc/34mBkr33)

### Model Results

[![image.png](https://i.postimg.cc/sDFRV06Q/image.png)](https://postimg.cc/z3kQ7p18)

[![image.png](https://i.postimg.cc/CxRw6VXt/image.png)](https://postimg.cc/HrCGVFp4)


[![image.png](https://i.postimg.cc/4NRrCHyp/image.png)](https://postimg.cc/yWnfcdFN)


[![image.png](https://i.postimg.cc/RV12vCJy/image.png)](https://postimg.cc/N5Md4c4k)


### References

* https://opensourcelibs.com/lib/finetune-detr
* https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb#scrollTo=h51Dd0udpfkG
* https://www.youtube.com/watch?v=T35ba_VXkMY
