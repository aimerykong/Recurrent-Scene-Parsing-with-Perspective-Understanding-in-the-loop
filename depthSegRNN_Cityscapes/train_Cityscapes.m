%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

%}
clear
close all
clc;

addpath(genpath('../libs'))
path_to_matconvnet = '../matconvnet';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU 
gpuId = 2; 
gpuDevice(gpuId);
%% load imdb file
load('imdb_cityscapes.mat');
imdb.meta.classNum = imdb.num_classes;

idfactor = 10.^([1,2,3]-1);
labelDict = load('labelDictionary.mat');
validIndex = find(labelDict.ignoreInEval==0);
colorLabel = labelDict.colorLabel(validIndex,:);
colorID = sum(bsxfun(@times, colorLabel, idfactor), 2);
categoryName = labelDict.categoryName(validIndex);
classID = labelDict.classID(validIndex);
className = labelDict.className(validIndex);
hasInstances = labelDict.hasInstances(validIndex);
trainClassID = labelDict.trainClassID(validIndex);
classNum = length(trainClassID);

if ~exist('legendRGB.jpg', 'file')
    legendRGB = zeros(400,200,3);
    for i = 0:classNum-1
        legendRGB(1+i*20:i*20+20,:,1) = colorLabel(i+1,1);
        legendRGB(1+i*20:i*20+20,:,2) = colorLabel(i+1,2);
        legendRGB(1+i*20:i*20+20,:,3) = colorLabel(i+1,3);
    end
    figure(1000);
    imshow(uint8(legendRGB)); %title('legend');
    for i = 1:19
        text(20, i*20-10, className{i}, 'rotation', 00, 'color', 'white', 'fontSize', 12);
    end
    text(20, 20*20-10, 'void', 'rotation', 00, 'color', 'white', 'fontSize', 12)
    export_fig( 'legendRGB.jpg' );
end
legendRGB = imread('legendRGB.jpg');
%% sepcify model
modelName = 'Cityscapes_softmax_net-epoch-7.mat';

netbasemodel = load( fullfile(path_to_model, modelName) );
netbasemodel = netbasemodel.net;
for i = 1:numel(netbasemodel.layers)            
    curLayerName = netbasemodel.layers(i).name;
    if ~isempty(strfind(curLayerName, 'bn'))
        netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'usingGlobal');
        netbasemodel.layers(i).block.bnorm_moment_type_trn = 'global';
        netbasemodel.layers(i).block.bnorm_moment_type_tst = 'global';        
    end
end 
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);

scalingFactor = 1;
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
netbasemodel.meta.normalization.imageSize = [1024, 2048, 3, 1];
netbasemodel.meta.normalization.border = [304, 1328]; % 720x720
netbasemodel.meta.normalization.stepSize = [76, 83];
% netbasemodel.meta.normalization.border = [128, 1152]; % 896x896
% netbasemodel.meta.normalization.stepSize = [64, 384];
%% start loop-3
lName = sprintf('round3_concatLayer');
netbasemodel.addLayer(lName, dagnn.Concat('dim', 3),  {'res6_relu', 'recurrentModule2_block3_dropout'}, lName);
sName = lName;

baseName = 'recurrentModule3_block1';
kernelSZ = [3 3 1024 512];
stride = 1;
pad = 1;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_block1_conv_f'});
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'batch';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_block1_bn' '_g'], ['recurrentModule1_block1_bn' '_b'], ['recurrentModule1_block1_bn' '_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;
%%  depth prediction branch for loop-3
baseName = 'recurrentModule3_depthEstLayerOne';  
kernelSZ = [3 3 512 128];
stride = 1;
pad = 1;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerOne' '_conv_f']});
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'batch';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerOne' '_bn_g'], ['recurrentModule1_depthEstLayerOne' '_bn_b'], ['recurrentModule1_depthEstLayerOne' '_bn_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


baseName = 'recurrentModule3_depthEstLayerTwo';  
kernelSZ = [3 3 128 32];
stride = 1;
pad = 1;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerTwo' '_conv_f']});
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'batch';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerTwo' '_bn_g'], ['recurrentModule1_depthEstLayerTwo' '_bn_b'], ['recurrentModule1_depthEstLayerTwo' '_bn_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = [baseName, '_dropout'];
block = dagnn.DropOut('rate', 0.1);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = 'recurrentModule3_depthCls_conv';
block = dagnn.Conv('size', [1 1 32 5], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthCls_conv' '_f'], ['recurrentModule1_depthCls_conv' '_b']});
sName = lName;

lName = 'recurrentModule3_depthSoftmax';
netbasemodel.addLayer(lName, dagnn.SoftMax(), sName, lName);

baseName = 'recurrentModule3_depthCls';
lName = [baseName, '_interp'];
upsample_fac = 8;
filters = single(bilinear_u(upsample_fac*2, 5, 5));
crop = ones(1,4) * upsample_fac/2;
netbasemodel.addLayer(lName, ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', upsample_fac, ...
    'crop', crop, ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', 5, ...
    'hasBias', false), ...
    sName, lName, {['recurrentModule1_depthCls'  '_interp_f']}) ;
sName = lName;

obj_name = 'recurrentModule3_depthEstSoftmaxloss';
gt_name =  'depthID';
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {sName, gt_name}, obj_name)
%% depth-aware pyramid atrous convolution for loop-2
for poolIdx = [1 2 4 8 16]  
    sName = 'recurrentModule3_block1_relu';
    baseName = sprintf('recurrentModule3_block2_pyramidAtrous_pool%d',poolIdx);
    kernelSZ = [3 3 512 512];
    stride = 1;
    pad = poolIdx;
    hasBias = false;
    dilate = poolIdx;
    
    lName = [baseName '_conv'];
    block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
    netbasemodel.addLayer(lName, block, sName, lName, {sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_conv_f',poolIdx)});
    sName = lName;
    lName = [baseName, '_bn'];
    block = dagnn.BatchNorm('numChannels', kernelSZ(end));
    block.bnorm_moment_type_trn = 'batch';
    block.bnorm_moment_type_tst = 'global';
    netbasemodel.addLayer(lName, block, sName, lName, {sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_g',poolIdx), sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_b',poolIdx), sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_m',poolIdx)});
    sName = lName;
    lName = [baseName, '_relu'];
    block = dagnn.ReLU('leak', 0);
    netbasemodel.addLayer(lName, block, sName, lName);
end


lName = 'recurrentModule3_depthGatingLayer';
block = dagnn.MaskGating();
netbasemodel.addLayer(lName, block,  {'recurrentModule3_block2_pyramidAtrous_pool1_relu', 'recurrentModule3_block2_pyramidAtrous_pool2_relu', ...
    'recurrentModule3_block2_pyramidAtrous_pool4_relu', 'recurrentModule3_block2_pyramidAtrous_pool8_relu', 'recurrentModule3_block2_pyramidAtrous_pool16_relu', ...
    'recurrentModule3_depthSoftmax'}, lName);
sName = lName;


baseName = 'recurrentModule3_block3';
kernelSZ = [1 1 512 512];
stride = 1;
pad = 0;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_block3_conv_f'});
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'batch';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_block3_bn' '_g'], ['recurrentModule1_block3_bn' '_b'], ['recurrentModule1_block3_bn' '_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = [baseName '_dropout'];
block = dagnn.DropOut('rate', 0.1);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = ['recurrentModule3' '_cls'];
block = dagnn.Conv('size', [1 1 512 imdb.meta.classNum], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1' '_cls' '_f'], ['recurrentModule1' '_cls' '_b']});
sName = lName;


lName = ['recurrentModule3', '_interp'];
upsample_fac = 8;
filters = single(bilinear_u(upsample_fac*2, 19, 19));
crop = ones(1,4) * upsample_fac/2;
netbasemodel.addLayer(lName, ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', upsample_fac, ...
    'crop', crop, ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', 19, ...
    'hasBias', false), ...
    sName, lName, {['recurrentModule1' '_interp_f']}) ;
ind = netbasemodel.getParamIndex([lName  '_f']) ;
sName = lName;

obj_name = sprintf('obj_Recurrent3');
gt_name =  sprintf('gt_div%d_seg', scalingFactor);
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {sName, gt_name}, obj_name)

%% set learning rate
netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_cls_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_cls_b')).learningRate = 20;
for i = 1:numel(netbasemodel.params)            
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);   
end  
%% configure training environment
batchSize = 1;
totalEpoch = 150;
learningRate = 1:totalEpoch;
learningRate = (2e-4) * (1-learningRate/totalEpoch).^0.9; %epoch35, increase learning rate
% learningRate = (5e-5) * (1-learningRate/totalEpoch).^0.9;


weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'train_Cityscapes_recurentRound3PredDepthAtrous_v1_allLoss');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end

opts.withDepth = false ;
opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = false ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.numEpochs = numel(opts.learningRate) ;
opts.learningRate = learningRate;

for i = 1:3
    curSetName = imdb.sets.name{i};
    curSetID = imdb.sets.id(i);
    curList = find(imdb.images.set==curSetID);
    opts.(curSetName) = curList(1:end);    
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getImgBatchWrapper(bopts); %  at epoch 123, rm train-aug

opts.backPropDepth = 10;%inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'recurrentModule1_block1_conv';

trainfn = @cnnTrain;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', {...
    'recurrentModule3_depthEstSoftmaxloss', 0.01, 'obj_Recurrent3', 1,...
    'recurrentModule2_depthEstSoftmaxloss', 0.01, 'obj_Recurrent2', 0.5,...
    'recurrentModule1_depthEstSoftmaxloss', 0.01, 'obj_Recurrent1', 0.1, 'obj_div1_seg', 0, 'depthEstSoftmaxloss', 0}, opts);
%% leaving blank
%{
./evalPixelLevelSemanticLabeling.py main010_recurentRound2PredDepthAtrous_v1_2loss_softmax_net-epoch-7_AVG_predEval/
classes          IoU      nIoU
--------------------------------
road          : 0.983      nan
sidewalk      : 0.859      nan
building      : 0.926      nan
wall          : 0.549      nan
fence         : 0.623      nan
pole          : 0.640      nan
traffic light : 0.711      nan
traffic sign  : 0.800      nan
vegetation    : 0.925      nan
terrain       : 0.637      nan
sky           : 0.947      nan
person        : 0.822    0.653
rider         : 0.631    0.484
car           : 0.950    0.877
truck         : 0.740    0.450
bus           : 0.837    0.633
train         : 0.754    0.555
motorcycle    : 0.645    0.466
bicycle       : 0.772    0.603
--------------------------------
Score Average : 0.776    0.590
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.986      nan
nature        : 0.927      nan
object        : 0.713      nan
sky           : 0.947      nan
construction  : 0.932      nan
human         : 0.834    0.679
vehicle       : 0.939    0.855
--------------------------------
Score Average : 0.897    0.767
--------------------------------



./evalPixelLevelSemanticLabeling.py main010_recurentRound2PredDepthAtrous_v1_2loss_softmax_net-epoch-7_MAX_predEval/
classes          IoU      nIoU
--------------------------------
road          : 0.983      nan
sidewalk      : 0.860      nan
building      : 0.926      nan
wall          : 0.549      nan
fence         : 0.623      nan
pole          : 0.641      nan
traffic light : 0.711      nan
traffic sign  : 0.800      nan
vegetation    : 0.925      nan
terrain       : 0.639      nan
sky           : 0.947      nan
person        : 0.822    0.656
rider         : 0.629    0.484
car           : 0.950    0.877
truck         : 0.737    0.451
bus           : 0.835    0.632
train         : 0.752    0.556
motorcycle    : 0.642    0.464
bicycle       : 0.772    0.601
--------------------------------
Score Average : 0.776    0.590
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.986      nan
nature        : 0.927      nan
object        : 0.713      nan
sky           : 0.947      nan
construction  : 0.932      nan
human         : 0.834    0.682
vehicle       : 0.939    0.855
--------------------------------
Score Average : 0.897    0.768
--------------------------------



./evalPixelLevelSemanticLabeling.py main010_recurentRound2PredDepthAtrous_v1_2loss_softmax_net-epoch-7_FFpath_predEval/
classes          IoU      nIoU
--------------------------------
road          : 0.982      nan
sidewalk      : 0.860      nan
building      : 0.926      nan
wall          : 0.546      nan
fence         : 0.628      nan
pole          : 0.638      nan
traffic light : 0.708      nan
traffic sign  : 0.796      nan
vegetation    : 0.924      nan
terrain       : 0.638      nan
sky           : 0.945      nan
person        : 0.817    0.663
rider         : 0.620    0.477
car           : 0.946    0.873
truck         : 0.699    0.430
bus           : 0.819    0.623
train         : 0.735    0.547
motorcycle    : 0.619    0.431
bicycle       : 0.766    0.595
--------------------------------
Score Average : 0.769    0.580
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.986      nan
nature        : 0.927      nan
object        : 0.710      nan
sky           : 0.945      nan
construction  : 0.932      nan
human         : 0.828    0.688
vehicle       : 0.936    0.851
--------------------------------
Score Average : 0.895    0.770
--------------------------------



./evalPixelLevelSemanticLabeling.py main010_recurentRound2PredDepthAtrous_v1_2loss_softmax_net-epoch-7_RCNN1_predEval/
classes          IoU      nIoU
--------------------------------
road          : 0.983      nan
sidewalk      : 0.857      nan
building      : 0.926      nan
wall          : 0.547      nan
fence         : 0.618      nan
pole          : 0.638      nan
traffic light : 0.706      nan
traffic sign  : 0.798      nan
vegetation    : 0.924      nan
terrain       : 0.631      nan
sky           : 0.946      nan
person        : 0.817    0.638
rider         : 0.630    0.485
car           : 0.950    0.875
truck         : 0.747    0.462
bus           : 0.836    0.633
train         : 0.751    0.546
motorcycle    : 0.649    0.473
bicycle       : 0.769    0.604
--------------------------------
Score Average : 0.775    0.590
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.986      nan
nature        : 0.926      nan
object        : 0.710      nan
sky           : 0.946      nan
construction  : 0.932      nan
human         : 0.829    0.667
vehicle       : 0.938    0.853
--------------------------------
Score Average : 0.895    0.760
--------------------------------


./evalPixelLevelSemanticLabeling.py main010_recurentRound2PredDepthAtrous_v1_2loss_softmax_net-epoch-7_RCNN2_predEval/
classes          IoU      nIoU
--------------------------------
road          : 0.982      nan
sidewalk      : 0.850      nan
building      : 0.924      nan
wall          : 0.540      nan
fence         : 0.606      nan
pole          : 0.627      nan
traffic light : 0.708      nan
traffic sign  : 0.795      nan
vegetation    : 0.923      nan
terrain       : 0.625      nan
sky           : 0.946      nan
person        : 0.812    0.633
rider         : 0.630    0.479
car           : 0.949    0.875
truck         : 0.755    0.452
bus           : 0.833    0.624
train         : 0.758    0.558
motorcycle    : 0.644    0.470
bicycle       : 0.769    0.597
--------------------------------
Score Average : 0.772    0.586
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.985      nan
nature        : 0.925      nan
object        : 0.702      nan
sky           : 0.946      nan
construction  : 0.931      nan
human         : 0.824    0.661
vehicle       : 0.937    0.852
--------------------------------
Score Average : 0.893    0.757
--------------------------------
%}

