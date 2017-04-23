# Recurrent-Scene-Parser-with-depthEstimation-in-the-loop


see http://www.ics.uci.edu/~skong2/recurrentDepthSeg


## cityscapes dataset
performance on *valset* [in training, fine-annotated trainset only, flip-augmentation only, one GPU, resnet101 as front-end chunk, softmax loss, multi-scale test]

classes | IoU | nIoU
-------------|-------------|-------------
**Score Average  | 0.791   |  0.617**
road           | 0.984   |    nan
sidewalk       | 0.866   |    nan
building       | 0.931   |    nan
wall           | 0.573   |    nan
fence          | 0.642   |    nan
pole           | 0.667   |    nan
traffic light  | 0.723   |    nan
traffic sign   | 0.816   |    nan
vegetation     | 0.929   |    nan
terrain        | 0.647   |    nan
sky            | 0.951   |    nan
person         | 0.834   |  0.678
rider          | 0.645   |  0.512
car            | 0.954   |  0.889
truck          | 0.777   |  0.476
bus            | 0.856   |  0.666
train          | 0.780   |  0.580
motorcycle     | 0.678   |  0.509
bicycle        | 0.780   |  0.624


04/22/2017
Shu Kong @ UCI
