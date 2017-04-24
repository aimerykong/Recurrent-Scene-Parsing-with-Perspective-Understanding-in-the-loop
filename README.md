# Recurrent Scene Parser with Perspective Understanding in the Loop


see [project page](http://www.ics.uci.edu/~skong2/recurrentDepthSeg)


## cityscapes dataset
performance on *valset* [in training, fine-annotated trainset only, flip-augmentation only, one GPU, resnet101 as front-end chunk, softmax loss, multi-scale test]


baseline (deeplab with single scale branch)
classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU | 0.738 | 0.980  |  0.849 |  0.916 | 0.475  | 0.596  | 0.598  | 0.684|  0.780 | 0.918 |  0.619  | 0.941  | 0.803  | 0.594  | 0.939  | 0.631  | 0.759  | 0.621   |  0.562 | 0.755    
nIoU |  0.547 | nan |    nan|     nan|   nan|   nan|   nan|     nan|     nan|    nan|    nan|   nan| 0.635| 0.448| 0.859| 0.398| 0.595| 0.467|  0.396|  0.582

ours-loop-2
classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU      | **`0.791`** | 0.984     | 0.866     | 0.931      | 0.573      | 0.642     | 0.667 | 0.723   |  0.816       | 0.929     | 0.647      | 0.951     | 0.834    | 0.645      | 0.954    | 0.777      | 0.856    | 0.780     | 0.678     | 0.780 
nIoU | **`0.617`** |     nan|    nan|    nan |    nan |    nan |    nan|  nan|   nan|    nan|    nan |    nan |  0.678|  0.512 |  0.889|  0.476 |  0.666|  0.580|  0.509|  0.624






nIoU

classes | IoU | nIoU
-------------|-------------|-------------
**`Score Average`**  | **`0.791`**   |  **`0.617`**
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
