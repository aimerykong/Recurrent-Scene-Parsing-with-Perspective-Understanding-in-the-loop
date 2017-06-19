% return a get batch function
% -------------------------------------------------------------------------
function fn = getImgBatchWrapper(opts)
% -------------------------------------------------------------------------
    fn = @(images,mode,scaleFactor) getBatch_dict(images, mode, scaleFactor, opts) ;
end

% -------------------------------------------------------------------------
function [im, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, dataMat, arrayGT_depthID, arrayGT_depthClass] = getBatch_dict(images, mode, scaleFactor, opts)
% -------------------------------------------------------------------------
    %images = strcat([imdb.path_to_dataset filesep], imdb.(mode).(batch) ) ; 
    [im, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, dataMat, arrayGT_depthID, arrayGT_depthClass] = getImgBatch(images, mode, scaleFactor, opts, ...
                                'prefetch', nargout == 0) ;
end
