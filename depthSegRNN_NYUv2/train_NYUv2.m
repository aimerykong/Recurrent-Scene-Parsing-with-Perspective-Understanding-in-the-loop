%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

credit: A few files are copied from Dr. Guosheng Lin's MatConvNet.

Shu Kong @ UCI
05/26/2017

%}
clear
close all
clc;

addpath(genpath('../libs'))
path_to_matconvnet = '../matconvnet';
path_to_model = '../models/';

load('NYUv2_label2color.mat');
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU 
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
%% load imdb and dataset file
load('imdb_NYUv2_offline.mat');
%% sepcify model
modelName = 'NYUv2_softmax_net-epoch-26.mat';

netbasemodel = load( fullfile(path_to_model, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
scalingFactor = 1;
%% modify the basemodel to fit segmentation task
% add objective function layer
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
netbasemodel.meta.normalization.imageSize = [imdb.meta.height, imdb.meta.width, 3, 1];
netbasemodel.meta.normalization.border = [25, 16]; % 720x720
netbasemodel.meta.normalization.stepSize = [1, 1];
%{
%% start loop-2
lName = sprintf('round2_concatLayer');
netbasemodel.addLayer(lName, dagnn.Concat('dim', 3),  {'res6_relu', 'recurrentModule1_block3_dropout'}, lName);
sName = lName;

baseName = 'recurrentModule2_block1';
kernelSZ = [3 3 1024 512];
stride = 1;
pad = 1;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_block1_conv_f'}); % size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_block1_conv_f')).value)
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'global';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_block1_bn' '_g'], ['recurrentModule1_block1_bn' '_b'], ['recurrentModule1_block1_bn' '_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;
%%  depth prediction branch for loop-3
baseName = 'recurrentModule2_depthEstLayerOne';  
kernelSZ = [1 1 512 128];
stride = 1;
pad = 0;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerOne_conv_f']}); % size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_depthEstLayerOne_conv_f')).value)
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'global';
block.bnorm_moment_type_tst = 'global';
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerOne' '_bn_g'], ['recurrentModule1_depthEstLayerOne' '_bn_b'], ['recurrentModule1_depthEstLayerOne' '_bn_m']});
sName = lName;
lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

baseName = 'recurrentModule2_depthEstLayerTwo';  
kernelSZ = [3 3 128 32];
stride = 1;
pad = 1;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1_depthEstLayerTwo_conv_f']}); % size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_depthEstLayerTwo_conv_f')).value)
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'global';
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

lName = 'recurrentModule2_depthCls_conv';
block = dagnn.Conv('size', [1 1 32 5], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_depthCls_f', 'recurrentModule1_depthCls_b'}); %  size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_depthCls_f')).value)
sName = lName;

lName = 'recurrentModule2_depthSoftmax';
netbasemodel.addLayer(lName, dagnn.SoftMax(), sName, lName);

obj_name = 'recurrentModule2_depthEstSoftmaxloss';
gt_name =  'depthID';
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {sName, gt_name}, obj_name)

sName = 'recurrentModule2_depthEstLayerTwo_relu';
lName = 'recurrentModule2_depthReg';
block = dagnn.Conv('size', [3 3 32 1], 'hasBias', true, 'stride', 1, 'pad', 1, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_depthReg_f', 'recurrentModule1_depthReg_b'});%  size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_depthReg_f')).value)
sName = lName;

obj_name = 'recurrentModule2_depthEstRegLoss';
gt_name =  'depthLogSpace';
netbasemodel.addLayer(obj_name, dagnn.Loss('loss', 'depthRegLoss'), {sName, gt_name}, obj_name);
%% depth-aware pyramid atrous convolution for loop-2
for poolIdx = [1 2 4 8 16]  
    sName = 'recurrentModule2_block1_relu';
    baseName = sprintf('recurrentModule2_block2_pyramidAtrous_pool%d',poolIdx);
    kernelSZ = [3 3 512 512];
    stride = 1;
    pad = poolIdx;
    hasBias = false;
    dilate = poolIdx;
    
    lName = [baseName '_conv'];
    block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
    netbasemodel.addLayer(lName, block, sName, lName, {sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_conv_f',poolIdx)});% size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_depthReg_f')).value)
    sName = lName;
    lName = [baseName, '_bn'];
    block = dagnn.BatchNorm('numChannels', kernelSZ(end));
    block.bnorm_moment_type_trn = 'global';
    block.bnorm_moment_type_tst = 'global';
    netbasemodel.addLayer(lName, block, sName, lName, {sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_g',poolIdx), sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_b',poolIdx), sprintf('recurrentModule1_block2_pyramidAtrous_pool%d_bn_m',poolIdx)});
    sName = lName;
    lName = [baseName, '_relu'];
    block = dagnn.ReLU('leak', 0);
    netbasemodel.addLayer(lName, block, sName, lName);
end

lName = 'recurrentModule2_depthGatingLayer';
block = dagnn.MaskGating();
netbasemodel.addLayer(lName, block,  {'recurrentModule2_block2_pyramidAtrous_pool16_relu', ...
    'recurrentModule2_block2_pyramidAtrous_pool8_relu', ...
    'recurrentModule2_block2_pyramidAtrous_pool4_relu', ...
    'recurrentModule2_block2_pyramidAtrous_pool2_relu', ...
    'recurrentModule2_block2_pyramidAtrous_pool1_relu', ...
    'recurrentModule2_depthSoftmax'}, lName);
sName = lName;

baseName = 'recurrentModule2_block3';
kernelSZ = [1 1 512 512];
stride = 1;
pad = 0;
hasBias = false;
dilate = 1;
lName = [baseName '_conv'];
block = dagnn.Conv('size', kernelSZ, 'hasBias', hasBias, 'stride', stride, 'pad', pad, 'dilate', dilate);
netbasemodel.addLayer(lName, block, sName, lName, {'recurrentModule1_block3_conv_f'}); % size(netbasemodel.params(netbasemodel.getParamIndex('recurrentModule1_block3_conv_f')).value)
sName = lName;
lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', kernelSZ(end));
block.bnorm_moment_type_trn = 'global';
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

lName = ['recurrentModule2' '_cls'];
block = dagnn.Conv('size', [1 1 512 imdb.meta.classNum], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {['recurrentModule1' '_cls' '_f'], ['recurrentModule1' '_cls' '_b']});
sName = lName;% size(netbasemodel.params(netbasemodel.getParamIndex(['recurrentModule1' '_cls' '_f'])).value)

obj_name = sprintf('obj_Recurrent2');
gt_name =  sprintf('gt_div%d_seg', scalingFactor);
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {sName, gt_name}, obj_name)
%}
%% set learning rate for the layers
for i = 1:numel(netbasemodel.layers)            
    curLayerName = netbasemodel.layers(i).name;
    if ~isempty(strfind(curLayerName, 'bn'))
        netbasemodel.layers(i).block.bnorm_moment_type_trn = 'global';
        netbasemodel.layers(i).block.bnorm_moment_type_tst = 'global';
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).learningRate = 0;
        fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).name, netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).learningRate);        
    end
end 
for i = 1:numel(netbasemodel.params)            
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);   
end  
%% configure training environment
batchSize = 1;
totalEpoch = 150;
learningRate = 1:totalEpoch;
learningRate = (1e-6) * (1-learningRate/totalEpoch).^0.9; 

weightDecay=0.0005; % weightDecay: usually use the default value
opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'train_it');
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

for i = 1:2
    curSetName = imdb.sets.name{i};
    opts.(curSetName) = imdb.(curSetName);
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
% fn = getImgBatchWrapper_NYUv2_offline(bopts);
fn = getImgBatchWrapper_NYUv2_offline_scaleAug(bopts);

% at epoch 33, remove trainaug

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
% opts.backPropAboveLayerName = 'res6_conv'; % 
opts.backPropAboveLayerName = 'res5_1_projBranch';% 
% opts.backPropAboveLayerName = 'res4_1_projBranch';% 

trainfn = @cnnTrainPredDepthGating_depthRegression;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, 'derOutputs', {...
    sprintf('obj_div%d_seg', scalingFactor), 1, 'depthEstSoftmaxloss', 0.01, 'depthEstRegLoss', 0.001, ...
    'obj_Recurrent1', 5, 'recurrentModule1_depthEstSoftmaxloss', 0.01, 'recurrentModule1_depthEstRegLoss', 0.001, ...
    'obj_Recurrent2', 5, 'recurrentModule2_depthEstSoftmaxloss', 0.01, 'recurrentModule2_depthEstRegLoss', 0.001}, ... % 
    opts);


%% leaving blank
%{

%}
