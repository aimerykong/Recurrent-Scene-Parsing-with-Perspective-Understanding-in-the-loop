% return a get batch function
% -------------------------------------------------------------------------
function fn = getImgBatchWrapper_NYUv2_offline_scaleAug(opts)
% -------------------------------------------------------------------------
    fn = @(images,mode,scaleFactor) getBatch_dict(images, mode, scaleFactor, opts) ;
end

% -------------------------------------------------------------------------
function [im, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, dataMat, arrayGT_depthID, arrayGT_depthClass, arrayGT_depthLogSpace] = getBatch_dict(images, mode, scaleFactor, opts)
% -------------------------------------------------------------------------
    %images = strcat([imdb.path_to_dataset filesep], imdb.(mode).(batch) ) ; 
    [im, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, dataMat, arrayGT_depthID, arrayGT_depthClass, arrayGT_depthLogSpace] = getImgBatch_NYUv2_offline_scaleAug(images, mode, scaleFactor, opts, ...
                                'prefetch', nargout == 0) ;
end
