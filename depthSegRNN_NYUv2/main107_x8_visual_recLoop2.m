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

load('NYUv2_label2color.mat');
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);

%% read matconvnet model
% set GPU
gpuId = 3; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

modelName = 'NYUv2_softmax_net-epoch-26.mat';
%% setup network
netMat = load( fullfile(path_to_model, modelName) );
netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);

depthBinNum = 5;
netMat = removeLossLayers(netMat);

saveFolder = ['visualization'];

netMat.move('gpu');
netMat.mode = 'test' ;
% netMat.mode = 'normal' ;
netMat.conserveMemory = 1;
%% test 
imgList = dir('./*jpg');

for imgIdx = 1:length(imgList)
    imOrg = single(imread(imgList(imgIdx).name));
    %% feed into the network
    fprintf('image-%03d %s ... ', imgIdx, imgList(imgIdx).name);
    imFeed = bsxfun(@minus, imOrg, mean_rgb);                    
    inputs = {'data', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    %% gather the output 
    SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);    
    SoftMaxLayer_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer_depth')).outputIndexes).value);
    depthReg_interp = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('depthReg_interp')).outputIndexes).value);
    depthReg = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('depthReg')).outputIndexes).value);
    
    SoftMaxLayerAtRecurrent1 = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent1')).outputIndexes).value);
    SoftMaxLayerAtRecurrent1_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent1_depth')).outputIndexes).value);
    recurrentModule1_depthReg_interp = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('recurrentModule1_depthReg_interp')).outputIndexes).value);
    recurrentModule1_depthReg = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('recurrentModule1_depthReg')).outputIndexes).value);    
    
    SoftMaxLayerAtRecurrent2 = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent2')).outputIndexes).value);
    SoftMaxLayerAtRecurrent2_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent2_depth')).outputIndexes).value);
    recurrentModule2_depthReg_interp = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('recurrentModule2_depthReg_interp')).outputIndexes).value);
    recurrentModule2_depthReg = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('recurrentModule2_depthReg')).outputIndexes).value);    
    
    depthReg_Manual = exp(depthReg);
    depthReg_Manual = imresize(depthReg_Manual,8);
    depthReg = exp(depthReg_interp);
    
    recurrentModule1_depthReg_Manual = exp(recurrentModule1_depthReg);
    recurrentModule1_depthReg_Manual = imresize(recurrentModule1_depthReg_Manual,8);
    recurrentModule1_depthReg = exp(recurrentModule1_depthReg_interp);
    
    recurrentModule2_depthReg_Manual = exp(recurrentModule2_depthReg);
    recurrentModule2_depthReg_Manual = imresize(recurrentModule2_depthReg_Manual,8);
    recurrentModule2_depthReg = exp(recurrentModule2_depthReg_interp);
    
    [~, predSeg] = max(SoftMaxLayer,[],3);
    [~, predDepthMap] = max(SoftMaxLayer_depth,[],3);
    [~, predSegCurLoop1] = max(SoftMaxLayerAtRecurrent1,[],3);
    [~, predDepthMapCurLoop1] = max(SoftMaxLayerAtRecurrent1_depth,[],3);
    [~, predSegCurLoop2] = max(SoftMaxLayerAtRecurrent2,[],3);
    [~, predDepthMapCurLoop2] = max(SoftMaxLayerAtRecurrent2_depth,[],3);
    
    fprintf(' done!\n');
    %% visualization    
    predSegColor = NYUv2_label2color.mask_cmap(predSeg(:)+1,:);
    predSegColor = reshape(predSegColor, [size(predSeg,1),size(predSeg,2),3]);
    predSegCurLoop1Color = NYUv2_label2color.mask_cmap(predSegCurLoop1(:)+1,:);
    predSegCurLoop1Color = reshape(predSegCurLoop1Color, [size(predSegCurLoop1,1),size(predSegCurLoop1,2),3]);      
    predSegCurLoop2Color = NYUv2_label2color.mask_cmap(predSegCurLoop2(:)+1,:);
    predSegCurLoop2Color = reshape(predSegCurLoop2Color, [size(predSegCurLoop2,1),size(predSegCurLoop2,2),3]);
    
    imgFig = figure;
    subWindowH = 4; 
    subWindowW = 6;
    set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
    windowID = 1;
    
    % ground-truth
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 6;
    imagesc(uint8(imOrg)); title(sprintf('image-%04d', imgIdx)); axis off image;
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(gtDepthOrg); title(sprintf('gtDepthOrg')); axis off image;
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     gtDepthOrgLog = gtDepthOrg;
%     gtDepthOrgLog(gtDepthOrgLog<1) = 1;
%     imagesc(log(gtDepthOrg) ); title(sprintf('gtDepthOrg log space')); axis off image;
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(gtDepthBin); title(sprintf('gtDepthBin'));  axis off image;  caxis([0 5])
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(gtSeg); title(sprintf('gtSeg'));  axis off image; caxis([0 40]) 
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(gtSegColor); title(sprintf('gtSegColor'));  axis off image; caxis([0 5]);     
       
    % result from the forward-pathway
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predDepthMap); title('predDepthMap');  axis off image; caxis([0 5])       
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(depthReg_interp); title('depthReg_interp log space', 'Interpreter', 'none');  axis off image; 
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(depthReg); title('depthReg');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(depthReg_Manual); title('depthReg_Manual', 'Interpreter', 'none');  axis off image;  
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSeg); title('predSeg');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSegColor); title('predSegColor');  axis off image;     
    
    % result from the recurrent loop-1
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predDepthMapCurLoop1); title('predDepthMapCurLoop1');  axis off image; caxis([0 5])       
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule1_depthReg_interp); title('recurrentModule1_depthReg_interp log space', 'Interpreter', 'none');  axis off image; 
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule1_depthReg); title('recurrentModule1_depthReg', 'Interpreter', 'none');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule1_depthReg_Manual); title('recurrentModule1_depthReg_Manual', 'Interpreter', 'none');  axis off image;  
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSegCurLoop1); title('predSegCurLoop1');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSegCurLoop1Color); title('predSegCurLoop1Color');  axis off image;    
    
    % result from the recurrent loop-2
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predDepthMapCurLoop2); title('predDepthMapCurLoop2');  axis off image; caxis([0 5])       
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule2_depthReg_interp); title('recurrentModule2_depthReg_interp log space', 'Interpreter', 'none');  axis off image; 
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule2_depthReg); title('recurrentModule2_depthReg', 'Interpreter', 'none');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(recurrentModule2_depthReg_Manual); title('recurrentModule2_depthReg_Manual', 'Interpreter', 'none');  axis off image;  
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSegCurLoop2); title('predSegCurLoop2');  axis off image;     
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSegCurLoop2Color); title('predSegCurLoop2Color');  axis off image;    
    
        
    if flagSaveFig && ~isdir(saveFolder)
        mkdir(saveFolder);
    end
    if flagSaveFig
        export_fig( sprintf('%s/visualization_%s.jpg', saveFolder,  imgList(imgIdx).name) );
    end
end
figure(101);
plotLegend4NYUv2(NYUv2_label2color);
% export_fig( sprintf('%s/legend.jpg', saveFolder) );
%% leaving blank

