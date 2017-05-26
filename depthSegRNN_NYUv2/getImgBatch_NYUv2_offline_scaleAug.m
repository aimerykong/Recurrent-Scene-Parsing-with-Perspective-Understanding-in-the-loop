function [imt, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, datumMat, arrayGT_depthID, arrayGT_depthClass, arrayGT_depthLogSpace] = getImgBatch_NYUv2_offline_scaleAug(images, mode, scaleFactor, varargin)
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

% global dataset

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
% validIdx = find(opts.imdb.classes.trainid(1:end-1) < 100);
% validIdx = unique(opts.imdb.meta.mapClass);


for img_i = 1:1%:numel(images)
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
    curMat = load(fullfile(opts.imdb.path_to_dataset, mode, images{img_i}));
    imgOrg = curMat.datamat.img;
    gtOrg = curMat.datamat.segGT;
    gtDepthOrg = curMat.datamat.depth; 
%     gtDepthBin = curMat.datamat.depthBin; 
            
%     figure(1);
%     subplot(2,2,1); imshow(uint8(imgOrg));
%     subplot(2,2,2); imagesc(gtOrg); axis off image; caxis([0, 40]);
%     subplot(2,2,3); imagesc(gtDepthOrg); axis off image;
%     subplot(2,2,4); imagesc(gtDepthOrg); axis off image; colorbar; caxis([0 5]);
    %% augmentation
    if strcmpi(mode, 'train') 
        if flag_flip
            %% flip augmentation
            imgOrg = fliplr(imgOrg);
            gtOrg = fliplr(gtOrg);
            gtDepthOrg = fliplr(gtDepthOrg);
%             gtDepthBin = fliplr(gtDepthBin);
        end        
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);              
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);    
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);
%         gtDepthBin = gtDepthBin(ystart:yend, xstart:xend,:);
        %% gamma augmentation
        imgOrg = imgOrg / 255;
        Z = gammaRange(1) + (gammaRange(2)-gammaRange(1)).*rand(1);
        gamma = log(0.5 + 1/sqrt(2)*Z) / log(0.5 - 1/sqrt(2)*Z);
        imgOrg = imgOrg.^gamma * 255;
        %% RGB jittering
        if rand(1)>0.3
            jitterRGB = rand(1,1,3)*0.4+0.8;
            imgOrg = bsxfun(@times, imgOrg, jitterRGB);            
        end
        %% random rotation
        if rand(1)>0.3
            rangeDegree = -15:1:15;
            angle = randsample(rangeDegree, 1);
            if angle~=0
                W = size(imgOrg,2);
                H = size(imgOrg,1);
                Hst = ceil(W*abs(sin(angle/180*pi)));
                Wst = ceil(H*abs(sin(angle/180*pi)));
                
                imgOrg = imrotate(imgOrg, angle, 'bicubic');
                gtOrg = imrotate(gtOrg, angle, 'nearest');
                gtDepthOrg = imrotate(gtDepthOrg, angle, 'bicubic');
                %             gtDepthBin = imrotate(gtDepthBin, angle, 'nearest');
                %         gtStack = imrotate(gtStack, angle, 'bicubic');
                
                imgOrg = imgOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                gtOrg = gtOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                gtDepthOrg = gtDepthOrg(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                %             gtDepthBin = gtDepthBin(max(1,Hst):end-Hst, max(1,Wst):end-Wst, :);
                %         gtStack = gtStack(Hst:end-Hst, Wst:end-Wst, :);
                imgOrg(imgOrg<0) = 0;
                imgOrg(imgOrg>255) = 255;
            end
        %% random scaling          
            sz = size(imgOrg); sz = sz(1:2);
            %         scaleFactorList = {[304,408], [320,432], [344,464], [360,488], [400,544], [440,600], [464,624], [480,656], [504,680], [520,704], [560,760], [600,816]};
            scaleFactorList = 0.4:0.01:2;
            scaleFactorList = randsample(scaleFactorList, 1);
            curRandScaleFactor = round(scaleFactorList*sz/8)*8;
            
            imgOrg = imresize(imgOrg, curRandScaleFactor);
            gtOrg = imresize(gtOrg, curRandScaleFactor, 'nearest');
            gtDepthOrg = imresize(gtDepthOrg, curRandScaleFactor, 'bicubic');
            gtDepthOrg = gtDepthOrg ./ scaleFactorList;
            %         gtDepthBin = imresize(gtDepthBin, curRandScaleFactor, 'nearest');
            %         gtStack = imresize(gtStack, curRandScaleFactor);
            
            mask = (gtOrg~=0);
        end
    elseif strcmpi(mode, 'val') 
        %% crop augmentation
        imgOrg = imgOrg(ystart:yend, xstart:xend,:);    
        gtOrg = gtOrg(ystart:yend, xstart:xend,:);                
        gtDepthOrg = gtDepthOrg(ystart:yend, xstart:xend,:);      
%         gtDepthBin = gtDepthBin(ystart:yend, xstart:xend,:);     
    end    
    %% others  
    threshList = [1, 1500, 2100, 3000, 4700, 100000];
    gtDepthBin = gtDepthOrg;
    for i = 1:length(threshList)-1
        gtDepthBin(gtDepthBin>threshList(i) & gtDepthBin<=threshList(i+1)) = i;
    end

%     mask = imresize(mask, 1/scaleFactor, 'nearest');
%     curGT_Stack = imresize(gtStack, 1/scaleFactor, 'bicubic');
%     [~, curGT_Stack] = max(curGT_Stack, [], 3);
%     curGT_Stack = bsxfun(@times, curGT_Stack, mask);

%     subplot(2,2,1); imshow(uint8(imgOrg));
%     subplot(2,2,2); imagesc(gtMap); axis off image; caxis([0, 19]);
%     subplot(2,2,3); imagesc(gtMap_div8); axis off image; caxis([0, 19]);
    
    gtDepthBinStack = zeros(size(gtDepthOrg,1), size(gtDepthOrg,2), depthBinNum);
    A = gtDepthBin;
    A(A==0) = 1/depthBinNum;
    for iii = 1:depthBinNum     
        B = A;
        B(B~=iii & B >=1) = 0;
        B(B==iii) = 1;
        gtDepthBinStack(:,:,iii) = B;
    end
%     imo(:,:,:,img_i) = imgOrg;
%     imt(:,:,:,img_i) = bsxfun(@minus, imo(:,:,:,img_i), opts.averageImage) ;   
%     arrayGT_id(:,:,:,img_i) = curGT_Sctack;      
%     arrayGT_depthID(:,:,:,img_i) = gtDepthOrg; 
%     arrayGT_depthClass(:,:,:,img_i) = gtDepthOrgStack;  


    %% return
    imo = imgOrg;
    imt = bsxfun(@minus, imo(:,:,:,img_i), opts.averageImage) ;   
    arrayGT_id = gtOrg;%curGT_Stack;      
    arrayGT_depthID = gtDepthBin; %gtDepthOrg; 
    arrayGT_depthClass = gtDepthBinStack;  
    arrayGT_depthLogSpace = gtDepthOrg;
    arrayGT_depthLogSpace(arrayGT_depthLogSpace>1) = log(arrayGT_depthLogSpace(arrayGT_depthLogSpace>1));
end
% finishFlag = true;
%% leaving blank


