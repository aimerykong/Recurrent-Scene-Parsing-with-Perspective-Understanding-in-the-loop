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


map2valid = zeros(1, 100);
map2valid(classID) = 1;


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
legendRGB = cat(2, legendRGB(1:201,:,:), legendRGB(202:end,:,:) );


mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% read matconvnet model
% set GPU
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

modelName = 'Cityscapes_softmax_net-epoch-7.mat';
%% setup network
netMat = load( fullfile(path_to_model, modelName) );
netMat = netMat.net;
for i = 1:numel(netMat.layers)            
    curLayerName = netMat.layers(i).name;
    if ~isempty(strfind(curLayerName, 'bn'))
        netMat.layers(i).block = rmfield(netMat.layers(i).block, 'usingGlobal');
        netMat.layers(i).block.bnorm_moment_type_trn = 'global';
        netMat.layers(i).block.bnorm_moment_type_tst = 'global';        
    end
end 
netMat = dagnn.DagNN.loadobj(netMat);

rmLayerName = 'obj_div1_seg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayer');
    netMat.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer  
    layerTop = sprintf('SoftMaxLayer_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);  
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

rmLayerName = 'obj_Recurrent1';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayerAtRecurrent1');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule1_depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayerAtRecurrent1_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

rmLayerName = 'obj_Recurrent2';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayerAtRecurrent2');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule2_depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayerAtRecurrent2_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

[~, saveFolder] = fileparts(modelName);
saveFolder = [strrep(saveFolder, '/', '') '_visualization'];

netMat.move('gpu');
netMat.mode = 'test' ;
netMat.conserveMemory = 1;
%% test res2
imgList = dir('./*png');

for imgIdx = 1:1%length(imgList) [[[!!! change here if like !!!]]]
    imOrg = single(imread(imgList(imgIdx).name));
    imFeed = single(imOrg); % convert to single precision
    imFeed = bsxfun(@minus, imFeed, mean_rgb);
    
%     curDepthName = curImgName;
%     curDepthName = strrep(curDepthName, 'cityscapes_imgFine', 'disparity');
%     curDepthName = strrep(curDepthName, 'leftImg8bit', 'disparity');
%     depthMap = single(imread(curDepthName));
%     gtDepthOrg = depthMap;
%     gtDepthOrg(depthMap<=10) = 1;
%     gtDepthOrg = log2(gtDepthOrg);    
%     threshList = [0, 11, 12.5, 13.2, 13.9, 20];
%     for i = 1:length(threshList)-1
%         gtDepthOrg(gtDepthOrg>threshList(i) & gtDepthOrg<=threshList(i+1)) = i;
%     end    
%         
%     path_to_GTfolder = strrep(curImgName, 'cityscapes_imgFine', 'gtFine');
%     path_to_GTfolder = strrep(path_to_GTfolder, 'leftImg8bit', 'gtFine_color');
%     gtColorMap = imread(path_to_GTfolder);        
%     
%     path_to_GTfolder = strrep(curImgName, 'cityscapes_imgFine', 'gtFine');
%     path_to_GTfolder = strrep(path_to_GTfolder, 'leftImg8bit', 'gtFine_labelIds');
%     mask = imread(path_to_GTfolder);     
%     mask = single(mask);
%     mask(mask>0) = map2valid(mask(mask>0));
%     mask = single(sum(single(gtColorMap), 3)~=0);
%     mask = sum(bsxfun(@times, single(gtColorMap), idfactor), 2);
    
    
    fprintf('image-%03d %s ... \n', imgIdx, imgList(imgIdx).name);
        
    inputs = {'data', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    %% get the outputs
    SoftMaxLayerAtRecurrent2 = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent2')).outputIndexes).value);  
    SoftMaxLayerAtRecurrent2_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent2_depth')).outputIndexes).value);  
    SoftMaxLayerAtRecurrent1 = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent1')).outputIndexes).value);  
    SoftMaxLayerAtRecurrent1_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayerAtRecurrent1_depth')).outputIndexes).value);   
    SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);  
    SoftMaxLayer_depth = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer_depth')).outputIndexes).value);
    
    SoftMaxLayerMerged = max( cat(4,SoftMaxLayerAtRecurrent2,SoftMaxLayerAtRecurrent1,SoftMaxLayer), [], 4 );
    SoftMaxLayerMerged_depth = max( cat(4,SoftMaxLayerAtRecurrent2_depth,SoftMaxLayerAtRecurrent1_depth,SoftMaxLayer_depth), [], 4 );
    
    
    [~, predMap] = max(SoftMaxLayer,[],3);
    [~, predDepthMap] = max(SoftMaxLayer_depth,[],3);
    
    [~, predMapRound1] = max(SoftMaxLayerAtRecurrent1,[],3);  
    [~, predDepthMap1] = max(SoftMaxLayerAtRecurrent1_depth,[],3);  
    
    [~, predMapRound2] = max(SoftMaxLayerAtRecurrent2,[],3);  
    [~, predDepthMap2] = max(SoftMaxLayerAtRecurrent2_depth,[],3);  
    
    [~, predMapMerged] = max(SoftMaxLayerMerged,[],3);  
    [~, predDepthMapMerged] = max(SoftMaxLayerMerged_depth,[],3);  
        
    
    [output_softmax, evalLabelMap] = index2RGBlabel(predMap-1, colorLabel, classID);
    [output_softmaxRound1, evalLabelMapRound1] = index2RGBlabel(predMapRound1-1, colorLabel, classID);
    [output_softmaxRound2, evalLabelMapRound2] = index2RGBlabel(predMapRound2-1, colorLabel, classID);
    [output_softmaxMerged, evalLabelMapMerged] = index2RGBlabel(predMapMerged-1, colorLabel, classID);
        
%     diffLoop1 = mask.*sum((output_softmax~=output_softmaxRound1), 3); 
%     diffLoop2 = mask.*sum((output_softmaxRound2~=output_softmaxRound1), 3);
%     diffMerge = mask.*sum((output_softmaxMerged~=output_softmaxRound2), 3);
%     
%     diffFFGT = mask.*sum((output_softmax~=gtColorMap), 3); 
%     diffLoop1GT = mask.*sum((output_softmaxRound1~=gtColorMap), 3); 
%     diffLoop2GT = mask.*sum((output_softmaxRound2~=gtColorMap), 3);
%     diffMergeG = mask.*sum((output_softmaxMerged~=gtColorMap), 3);
%     
%     improveLoop1 = mask.*(sum((output_softmax~=gtColorMap),3) & sum((output_softmaxRound1==gtColorMap),3) );
%     improveLoop2 = mask.*(sum((output_softmaxRound1~=gtColorMap),3) & sum((output_softmaxRound2==gtColorMap),3) );
%     improveMerge = mask.*(sum((output_softmaxRound2~=gtColorMap),3) & sum((output_softmaxMerged==gtColorMap),3) );
%     
%     degradeLoop1 = mask.*(sum((output_softmax==gtColorMap),3) & sum((output_softmaxRound1~=gtColorMap),3) );
%     degradeLoop2 = mask.*(sum((output_softmaxRound1==gtColorMap),3) & sum((output_softmaxRound2~=gtColorMap),3) );
%     degradeMerge = mask.*(sum((output_softmaxRound2==gtColorMap),3) & sum((output_softmaxMerged~=gtColorMap),3) );   
    
    %% visualization
%     saveFolder11 = 'figure4paper_cityscapes';
%     if ~isdir(saveFolder11) 
%         mkdir(saveFolder11);
%     end
%     imwrite(uint8(imOrg), sprintf('%s/ImgId%04d_org.bmp', saveFolder11, testImgIdx) );
%     imwrite(uint8(gtColorMap), sprintf('%s/ImgId%04d_gtColorMap.bmp', saveFolder11, testImgIdx) );
%     imwrite(uint8(output_softmax), sprintf('%s/ImgId%04d_output_softmax.bmp', saveFolder11, testImgIdx) );
%     imwrite(uint8(output_softmaxRound1), sprintf('%s/ImgId%04d_output_softmaxRound1.bmp', saveFolder11, testImgIdx) );
%     imwrite(uint8(output_softmaxRound2), sprintf('%s/ImgId%04d_output_softmaxRound2.bmp', saveFolder11, testImgIdx) );
%     imwrite(uint8(output_softmaxMerged), sprintf('%s/ImgId%04d_output_softmaxMerged.bmp', saveFolder11, testImgIdx) );
    
    
    imgFig1 = figure(1);
    windowH = 2; windowW = 3;
    set(imgFig1, 'Position', [100 100 1800 700]) % [1 1 width height]    
    subplot(windowH,windowW,1); imagesc(uint8(imOrg)); title(sprintf('image-%04d', imgIdx)); axis off image; 
%     subplot(windowH,windowW,2); imagesc(uint8(gtColorMap)); title('gtColorMap'); axis off image; 
    subplot(windowH,windowW,3); imagesc(uint8(output_softmax)); title(sprintf('output_softmax'), 'Interpreter', 'none'); axis off image;
    subplot(windowH,windowW,4); imagesc(uint8(output_softmaxRound1)); title(sprintf('output_softmaxRound1'), 'Interpreter', 'none'); axis off image;
    subplot(windowH,windowW,5); imagesc(uint8(output_softmaxRound2)); title(sprintf('output_softmaxRound2'), 'Interpreter', 'none'); axis off image;
    subplot(windowH,windowW,6); imagesc(uint8(output_softmaxMerged)); title(sprintf('output_softmaxMerged'), 'Interpreter', 'none'); axis off image;
%     if flagSaveFig
%         export_fig( sprintf('%s/valImgId%04d_1_seg.jpg', saveFolder, testImgIdx) );
%     end
    

    imgFig2 = figure(2);
    windowH = 2; windowW = 3;
    set(imgFig2, 'Position', [100 100 1800 700]) % [1 1 width height]
    
%     subplot(windowH,windowW,1); imagesc(depthMap); axis off image;    
%     export_fig( sprintf('%s/ImgId%04d_depthMap.bmp', saveFolder11, testImgIdx)  );    
%     subplot(windowH,windowW,2); imagesc(gtDepthOrg); axis off image; caxis([0 5]);    
%     export_fig( sprintf('%s/ImgId%04d_gtDepthOrg.bmp', saveFolder11, testImgIdx)  );    
    subplot(windowH,windowW,3); imagesc(predDepthMap); axis off image; caxis([0 5]);
%     export_fig( sprintf('%s/ImgId%04d_predDepthMap.bmp', saveFolder11, testImgIdx)  );    
    subplot(windowH,windowW,4); imagesc(predDepthMap1); axis off image; caxis([0 5]);
%     export_fig( sprintf('%s/ImgId%04d_predDepthMap1.bmp', saveFolder11, testImgIdx)  );    
    subplot(windowH,windowW,5); imagesc(predDepthMap2); axis off image; caxis([0 5]);
%     export_fig( sprintf('%s/ImgId%04d_predDepthMap2.bmp', saveFolder11, testImgIdx)  );    
    subplot(windowH,windowW,6); imagesc(predDepthMapMerged); axis off image; caxis([0 5]);
%     export_fig( sprintf('%s/ImgId%04d_predDepthMapMerged.bmp', saveFolder11, testImgIdx)  );
    
    
%     if flagSaveFig
%         export_fig( sprintf('%s/valImgId%04d_2_depth.jpg', saveFolder, testImgIdx) );
%     end
    
%     imgFig3 = figure(3);
%     windowH = 3; windowW = 3;
%     set(imgFig3, 'Position', [100 100 1800 700]) % [1 1 width height]
%     subplot(windowH,windowW,1); imagesc(uint8(gtColorMap)); title(sprintf('gtColorMap')); axis off image;
%     subplot(windowH,windowW,2); imagesc(mask); title(sprintf('mask')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,3); imagesc(diffLoop1); title(sprintf('diffLoop1')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,4); imagesc(diffLoop2); title(sprintf('diffLoop2')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,5); imagesc(diffMerge); title(sprintf('diffMerge')); axis off image; caxis([0 1]); 
%     subplot(windowH,windowW,6); imagesc(diffFFGT); title(sprintf('diffFFGT')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,7); imagesc(diffLoop1GT); title(sprintf('diffLoop1GT')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,8); imagesc(diffLoop2GT); title(sprintf('diffLoop2GT')); axis off image; caxis([0 1]); 
%     subplot(windowH,windowW,9); imagesc(diffMergeG); title(sprintf('diffMergeG')); axis off image; caxis([0 1]); 
%     if flagSaveFig
%         export_fig( sprintf('%s/valImgId%04d_3_predDiff.jpg', saveFolder, testImgIdx) );
%     end
%     
%     
%     imgFig4 = figure(4);
%     windowH = 2; windowW = 3;
%     set(imgFig4, 'Position', [100 100 1800 700]) % [1 1 width height]
%     subplot(windowH,windowW,1); imagesc(improveLoop1); title(sprintf('improveLoop1')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,2); imagesc(improveLoop2); title(sprintf('improveLoop2')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,3); imagesc(improveMerge); title(sprintf('improveMerge')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,4); imagesc(degradeLoop1); title(sprintf('degradeLoop1')); axis off image; caxis([0 1]);
%     subplot(windowH,windowW,5); imagesc(degradeLoop2); title(sprintf('degradeLoop2')); axis off image; caxis([0 1]); 
%     subplot(windowH,windowW,6); imagesc(degradeMerge); title(sprintf('degradeMerge')); axis off image; caxis([0 1]);
%     if flagSaveFig
%         export_fig( sprintf('%s/valImgId%04d_4_quality.jpg', saveFolder, testImgIdx) );
%     end
    
    fprintf(' done!\n');
end
%% leaving blank

