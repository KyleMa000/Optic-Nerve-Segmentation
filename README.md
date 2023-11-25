# ReadMe Template
A real time face mask detector based on fine-tuned Faster R-CNN model.  
We fine-tuned the Faster R-CNN model trained on the COCO dataset, with two versions of using `MobileNetV3` and `ResNet50` as the backbone.  
Our models are able to detect `mask wear correcly`, `mask wear incorrectly`, and `no mask wear` in real time.  
We also made a webapp to give a demo of our model - `MobileNetV3`. Please see details at `./webapp`.  
[Presentation slides (Google slides)](https://docs.google.com/presentation/d/1tPoN7s3mpW8t46KBdZfxxvgLQ7JelQj9JMRtfdnAy0c/edit?usp=drivesdk)  

## Code structure
```

├── datasets/ # pytorch Dataset for the kaggle face mask dataset
├── utils/
|   ├── eval_utils.py # the utils for evaluating mAP
|   ├── model_utils.py # the utils for loading model
├── webapp/ # the directory for the webapp
|   ├── app.py # the webapp
|   ...
├── train.py # the training script
├── eval.py # the evaluation script for mAP
├── predict.py # make a single image prediction and draw the bounding box
├── inference_time.py # the inference time script
├── train_resnet50.sbatch # the SLURM script to train resnet50 on NYU HPC
|   ...
```
## Data preparation  
Download the dataset from [Face Mask Detection | Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Environment setup
1. Make sure your python version is `3.7`  
2. Install dependencies with `pip`: 
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --mode [mobilenetv3 / resnet50] --epoch 100 --logpath ./logs --dataroot /path/to/facemask/dataset
```

## Evaluation
Get mAP performace  
```bash
python eval.py --mode [mobilenetv3 / resnet50] --epoch 99 --dataroot /path/to/facemask/dataset
```

Get inference time
```bash
python inference_time.py --mode [mobilenetv3 / resnet50] --epoch 99 --dataroot /path/to/facemask/dataset
```

## Evaluation results
**MobilNetV3**  
- FPS: 24
- mAP: 0.378  

| AP @IoU | Score |
| ------- | ----- |
| 0.50    | 0.571 |
| 0.55    | 0.561 |
| 0.60    | 0.553 |
| 0.65    | 0.534 |
| 0.70    | 0.508 |
| 0.75    | 0.447 |
| 0.80    | 0.349 |
| 0.85    | 0.202 |
| 0.90    | 0.056 |
| 0.95    | 0.003 |

**ResNet50**  
- FPS: 9
- mAP: 0.384

| AP @IoU | Score |
| ------- | ----- |
| 0.50    | 0.573 |
| 0.55    | 0.569 |
| 0.60    | 0.568 |
| 0.65    | 0.564 |
| 0.70    | 0.529 |
| 0.75    | 0.474 |
| 0.80    | 0.355 |
| 0.85    | 0.184 |
| 0.90    | 0.028 |
| 0.95    | 0.001 |

Observation:  
ResNet50 has a slightly better mAP than MobilNetV3.  
However, it is 3x slower than MobilNetV3 during inference.  
Since there is not a marginal difference of the mAP between `MobilNetV3` and `ResNet50`, we prefer `MobilNetV3`.
## Webapp
check `./webapp` for the documentation of web app.

## Acknowledgement
Transfer learning: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html  
Training code reference: https://www.kaggle.com/code/daniel601/pytorch-fasterrcnn  
mAP code reference: https://www.kaggle.com/code/franciscop9/facemaskdetection-pytorch-faster-r-cnn   
