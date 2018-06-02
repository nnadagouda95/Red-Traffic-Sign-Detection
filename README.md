#Red-Sign-Traffic-Detection

In this project, the Tensorflow Object Detection API has been used to detect and classify the red signs on the German Traffic Sign Dtection Benchmark (GTSDB) dataset.

Object Detection API
https://github.com/tensorflow/models/tree/master/research/object_detection

Dataset
http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset

Two COCO-trained models were used to re-train on the GTSDB dataset

ssd_inception_v2_coco
faster_rcnn_inception_v2_coco

Tensorflow Object Detection installation

Follow the steps on the link mentioned above for detailed instruction on the installation

Data Pre-processing program
data_processing.ipynb

Commands to run training

Faster RCNN
python train.py --logtostderr --train_dir=training/model_faster_rcnn --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

SSD
python train.py --logtostderr --train_dir=training/model_ssd --pipeline_config_path=training/ssd_inception_v2_coco.config


Commands to run evaluation

Faster RCNN
python eval.py \ --logtostderr  --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config    --checkpoint_dir=training/model_faster_rcnn --eval_dir=eval/eval_faster_rcnn

SSD
python eval.py \ --logtostderr  --pipeline_config_path=training/ssd_inception_v2_coco.config    --checkpoint_dir=training/model_ssd --eval_dir=eval/eval_ssd

Running the Tensorboard

Training and evaluation results can be visualized in tensorboard
tensorboard --logdir=${training path}
