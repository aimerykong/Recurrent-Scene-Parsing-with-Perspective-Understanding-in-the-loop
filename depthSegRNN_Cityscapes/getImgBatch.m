function [imt, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, datumMat, arrayGT_depthID, arrayGT_depthClass] = getImgBatch(images, mode, scaleFactor, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [1024, 2048] ;
opts.border = [32, 32] ;
opts.stepSize = [32, 32] ;
opts.lambda = 1 ;
opts.keepAspect = true ;
opts.numAugments = 1 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = [] ;
% opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.classNum = 19;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.imdb = [];

opts = vl_argparse(opts, varargin);

depthBinNum = 5;
gammaRange = [-0.03, 0.03];
%% read mat (hdf5) file for image and label
arrayGT_class = [];
arrayEdge = [];
arrayMask = [];
arrayGT_color = [];
datumMat = [];
if strcmpi(mode, 'train') || strcmpi(mode, 'val')
    imt = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
    arrayGT_id = zeros((opts.imageSize(1)-opts.border(1))/scaleFactor, (opts.imageSize(2)-opts.border(2))/scaleFactor, 1, numel(images), 'single') ;
    arrayGT_depthID = zeros((opts.imageSize(1)-opts.border(1))/scaleFactor, (opts.imageSize(2)-opts.border(2))/scaleFactor, 1, numel(images), 'single') ;
    imo = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
else
    imt = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(images), 'single') ;
    arrayGT_id = zeros(opts.imageSize(1)/scaleFactor, opts.imageSize(2)/scaleFactor, 1, numel(images), 'single') ;
    arrayGT_depthID = zeros(opts.imageSize(1)/scaleFactor, opts.imageSize(2)/scaleFactor, 1, numel(images), 'single') ;
end
arrayGT_depthClass = zeros(size(imt,1), size(imt,2), depthBinNum);
validIdx = find(opts.imdb.classes.trainid(1:end-1) < 100);

for img_i = 1:numel(images)
    if  strcmpi(mode, 'val')        
        flag_flip = 0;
        xstart = 1;
        ystart = 1;
        
        xend = opts.imageSize(2) - (opts.border(2) - xstart+1);
        yend = opts.imageSize(1) - (opts.border(1) - ystart+1);
    else
        flag_flip = rand(1)>0.5;
        xstart = randperm(opts.border(2) / opts.stepSize(2) + 1,1)*opts.stepSize(2) - opts.stepSize(2) + 1;
        ystart = randperm(opts.border(1) / opts.stepSize(1) + 1,1)*opts.stepSize(1) - opts.stepSize(1) + 1;
        
        xend = opts.imageSize(2) - (opts.border(2) - xstart+1);
        yend = opts.imageSize(1) - (opts.border(1) - ystart+1);
    end
    %% read the image and annotation    
    curImgName = sprintf(opts.imdb.img_path, mode, opts.imdb.images.city{images(img_i)}, opts.imdb.images.name{images(img_i)});
    imgOrg = imread(curImgName);
    imgOrg = single(imgOrg);
    
    curGTName = sprintf(opts.imdb.anno_path, 'gtFine', mode, opts.imdb.images.city{images(img_i)}, opts.imdb.images.filename{images(img_i)});
    gtOrg = imread(curGTName);
    gtOrg = single(gtOrg);

    curGTDepthName = sprintf(opts.imdb.depth_path, mode, opts.imdb.images.city{images(img_i)}, opts.imdb.images.name{images(img_i)});
    gtDepthOrg = imread(curGTDepthName);
    gtDepthOrg = single(gtDepthOrg);
    gtDepthOrg(gtDepthOrg<=10) = 1;
    gtDepthOrg = log2(gtDepthOrg);    
    threshList = [0, 11, 12.5, 13.2, 13.9, 20];
    for i = 1:length(threshList)-1
        gtDepthOrg(gtDepthOrg>threshList(i) & gtDepthOrg<=threshList(i+1)) = i;
    end
    %% augmentation
    if strcmpi(mode, 'train') 
        if flag_flip
            %% flip augmentation
            imgOrg = fliplr(imgOrg);
            gtOrg = fliplr(gtOrg);
            gtDepthOrg = fliplr(gtDepthOrg);
        end        
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);              
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);    
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);
        %% gamma augmentation
        imgOrg = imgOrg / 255;
        Z = gammaRange(1) + (gammaRange(2)-gammaRange(1)).*rand(1);
        gamma = log(0.5 + 1/sqrt(2)*Z) / log(0.5 - 1/sqrt(2)*Z);
        imgOrg = imgOrg.^gamma * 255;
    elseif strcmpi(mode, 'val') 
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);              
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);                
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);     
    end    
    %% resize
    gtMap = zeros(size(gtOrg), 'single');
    gtStack = zeros(size(gtOrg,1), size(gtOrg,2), opts.imdb.num_classes, 'single');
    for classIdx = 1:length(validIdx)
        tmp = gtStack(:,:,classIdx);
        tmp(gtOrg==opts.imdb.classes.id(validIdx(classIdx))) = 1;
        gtStack(:,:,classIdx) = tmp;
        gtMap(gtOrg==opts.imdb.classes.id(validIdx(classIdx))) = opts.imdb.classes.trainid(validIdx(classIdx))+1;
    end
    mask = (gtMap~=0);
    mask = imresize(mask, 1/scaleFactor, 'nearest');
    curGT_Stack = imresize(gtStack, 1/scaleFactor, 'bicubic');
    [~, curGT_Stack] = max(curGT_Stack, [], 3);
    curGT_Stack = bsxfun(@times, curGT_Stack, mask);

%     subplot(2,2,1); imshow(uint8(imgOrg));
%     subplot(2,2,2); imagesc(gtMap); axis off image; caxis([0, 19]);
%     subplot(2,2,3); imagesc(gtMap_div8); axis off image; caxis([0, 19]);
    
    imo(:,:,:,img_i) = imgOrg;
    imt(:,:,:,img_i) = bsxfun(@minus, imo(:,:,:,img_i), opts.averageImage) ;   
    arrayGT_id(:,:,:,img_i) = curGT_Stack;      
    arrayGT_depthID(:,:,:,img_i) = gtDepthOrg; 
    
    gtDepthOrgStack = zeros(size(gtDepthOrg,1), size(gtDepthOrg,2), depthBinNum);
    A = gtDepthOrg;
    A(A==0) = 1/depthBinNum;
    for iii = 1:depthBinNum     
        B = A;
        B(B~=iii & B >=1) = 0;
        B(B==iii) = 1;
        gtDepthOrgStack(:,:,iii) = B;
    end
    arrayGT_depthClass(:,:,:,img_i) = gtDepthOrgStack;      
end
% finishFlag = true;
%% leaving blank


