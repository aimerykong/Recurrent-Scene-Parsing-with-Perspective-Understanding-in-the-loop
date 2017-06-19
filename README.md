# Recurrent Scene Parsing with Perspective Understanding in the Loop

![alt text](http://www.ics.uci.edu/~skong2/img/figure4paper_cityscapes.png "visualization")

Objects may appear at arbitrary scales in perspective images of a scene, posing a challenge for recognition systems that process an image at a fixed resolution. We propose a depth-aware gating module that adaptively chooses the pooling field size in a convolutional network architecture according to the object scale (inversely proportional to the depth) so that small details can be preserved for objects at distance and a larger receptive field can be used for objects nearer to the camera. The depth gating signal is provided from stereo disparity (when available) or estimated directly from a single image. We integrate this depth-aware gating into a recurrent convolutional neural network trained in an end-to-end fashion to perform semantic segmentation. Our recurrent module iteratively refines the segmentation results, leveraging the depth estimate and output prediction from the previous loop. Through extensive experiments on three popular large-scale RGB-D datasets, we demonstrate our approach achieves competitive semantic segmentation performance using more compact model than existing methods. Interestingly, we find segmentation performance improves when we estimate depth directly from the image rather than using "ground-truth" and the model produces state-of-the-art results for quantitative depth estimation from a single image. 

For details, please refer to our [project page](http://www.ics.uci.edu/~skong2/recurrentDepthSeg)

To download our models, please go [google drive](https://drive.google.com/open?id=0BxeylfSgpk1MaVlNZV96eVVqdWM) and put the models in directory 'models'.

MatConvNet is used in our project, some functions are changed. So it might be required to re-compile. Useful commands are --

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



If you find the code useful, please city our work
    @article{kong2017depthsegRNN,
      title={Recurrent Scene Parsing with Perspective Understanding in the Loop},
      author={Kong, Shu and Fowlkes, Charless},
      journal={arXiv preprint arXiv:1705.07238},
      year={2017}
    }




05/24/2017
Shu Kong @ UCI
