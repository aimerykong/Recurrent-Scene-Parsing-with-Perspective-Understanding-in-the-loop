# Recurrent Scene Parsing with Perspective Understanding in the Loop

![alt text](http://www.ics.uci.edu/~skong2/img/figure4paper_cityscapes.png "visualization")

Objects may appear at arbitrary scales in perspective images of a scene, posing a challenge for recognition systems that process an image at a fixed resolution. We propose a depth-aware gating module that adaptively chooses the pooling field size in a convolutional network architecture according to the object scale (inversely proportional to the depth) so that small details can be preserved for objects at distance and a larger receptive field can be used for objects nearer to the camera. The depth gating signal is provided from stereo disparity (when available) or estimated directly from a single image. We integrate this depth-aware gating into a recurrent convolutional neural network trained in an end-to-end fashion to perform semantic segmentation. Our recurrent module iteratively refines the segmentation results, leveraging the depth estimate and output prediction from the previous loop. Through extensive experiments on three popular large-scale RGB-D datasets, we demonstrate our approach achieves competitive semantic segmentation performance using more compact model than existing methods. Interestingly, we find segmentation performance improves when we estimate depth directly from the image rather than using "ground-truth" and the model produces state-of-the-art results for quantitative depth estimation from a single image. 

For details, please refer to our [project page](http://www.ics.uci.edu/~skong2/recurrentDepthSeg).

To download our models, please go [google drive](https://drive.google.com/open?id=0BxeylfSgpk1MaVlNZV96eVVqdWM) and put the models in directory 'models'.

Script [demo_Cityscapes.m](https://github.com/aimerykong/Recurrent-Scene-Parsing-with-Perspective-Understanding-in-the-loop/blob/master/depthSegRNN_Cityscapes/demo_Cityscapes.m) provides a demonstration to visualize the results. There are many comment lines as the ground-truth depth and annotation are not put here. When they are available with the correct path, one can uncomment to run and get the complete figures. Script [train_Cityscapes.m](https://github.com/aimerykong/Recurrent-Scene-Parsing-with-Perspective-Understanding-in-the-loop/blob/master/depthSegRNN_Cityscapes/train_Cityscapes.m) is used for training the model. Here, as an example, it loads the model used for demostration, and adds the third loop to fine-tune. To train it, you need to download the Cityscapes dataset, and modify the paths in imdb accordingly which is used to point to the images, annotations and depth maps.

```python
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab

path_to_matconvnet = '../matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

```

If you find the code useful, please cite our work

    @article{kong2017depthsegRNN,
      title={Recurrent Scene Parsing with Perspective Understanding in the Loop},
      author={Kong, Shu and Fowlkes, Charless},
      journal={arXiv preprint arXiv:1705.07238},
      year={2017}
    }


## Cityscapes dataset
performance on *valset* [in training, fine-annotated trainset only, flip-augmentation only, one GPU, resnet101 as front-end chunk, softmax loss, test w/o augmentation unless specified]


#### baseline (deeplab with single scale branch)

classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU | **`0.738`** | 0.980  |  0.849 |  0.916 | 0.475  | 0.596  | 0.598  | 0.684|  0.780 | 0.918 |  0.619  | 0.941  | 0.803  | 0.594  | 0.939  | 0.631  | 0.759  | 0.621   |  0.562 | 0.755    
nIoU |  **`0.547`** | nan |    nan|     nan|   nan|   nan|   nan|     nan|     nan|    nan|    nan|   nan| 0.635| 0.448| 0.859| 0.398| 0.595| 0.467|  0.396|  0.582

#### baseline + perspective estimation

classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU | **`0.748`** | 0.981 | 0.849 | 0.918|0.506 |0.605 |0.604 | 0.67| 0.775| 0.918| 0.627|0.940 |0.804 |0.602 |0.942 |0.679 |0.787 |0.656 | 0.591| 0.753|
nIoU |**`0.556`** |   nan|   nan|   nan|  nan|  nan|  nan|    nan|    nan|   nan|   nan|  nan|0.639|0.460|0.863|0.407|0.612|0.489| 0.398|0.575|


#### RecurrentLoop-1 w/ perspective estimation

classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU | **`0.772`**  |0.983  | 0.859 | 0.925 |0.527  |0.628  |0.640  | 0.703| 0.795 | 0.923 | 0.630  |0.946  |0.816  |0.620  |0.950  |0.748  |0.839  |0.753  | 0.626 | 0.764  |
nIoU | **`0.589`**|   nan|   nan|   nan|  nan|  nan|  nan|    nan|    nan|   nan|   nan|  nan|0.660|0.483|0.869|0.446|0.632|0.572| 0.452| 0.597|


#### RecurrentLoop-2 w/ perspective estimation

classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU | **`0.776`**|0.983  | 0.860  | 0.926  |0.549   |0.623  |0.641   | 0.711 | 0.800 | 0.925 | 0.639  |0.947   |0.822  |0.629  |0.950   |0.737  |0.835   |0.752  | 0.642 | 0.772  |
nIoU | **`0.590`**|  nan|   nan|   nan|  nan|  nan|  nan|    nan|    nan|   nan|   nan|  nan|0.656|0.484|0.877|0.451|0.632|0.556| 0.464| 0.601|

#### RecurrentLoop-2 w/ perspective estimation (multi-scale when testing)

classes | **`Score Average`** | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
IoU      | **`0.791`** | 0.984     | 0.866     | 0.931      | 0.573      | 0.642     | 0.667 | 0.723   |  0.816       | 0.929     | 0.647      | 0.951     | 0.834    | 0.645      | 0.954    | 0.777      | 0.856    | 0.780     | 0.678     | 0.780 
nIoU | **`0.617`** |     nan|    nan|    nan |    nan |    nan |    nan|  nan|   nan|    nan|    nan |    nan |  0.678|  0.512 |  0.889|  0.476 |  0.666|  0.580|  0.509|  0.624






06/19/2017
Shu Kong @ UCI
