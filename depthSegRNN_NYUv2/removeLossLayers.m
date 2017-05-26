function netMat = removeLossLayers(netMat)
depthBinNum = 5;

rmLayerName = 'obj_div1_seg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'res7';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, netMat.layers(netMat.getLayerIndex(sName)).block.size(end), netMat.layers(netMat.getLayerIndex(sName)).block.size(end)));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', netMat.layers(netMat.getLayerIndex(sName)).block.size(end), ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayer');
    netMat.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'depthCls';
    lName = [baseName, '_interp'];
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 5, 5));
    crop = ones(1,4) * upsample_fac/2;
    netMat.addLayer(lName, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 5, ...
        'hasBias', false), ...
        sName, lName, {[lName  '_f']}) ;
    ind = netMat.getParamIndex([lName  '_f']) ;
    netMat.params(ind).value = filters ;
    netMat.params(ind).learningRate = 0 ;
    netMat.params(ind).weightDecay = 1 ;
    sName = lName;
    
    layerTop = sprintf('SoftMaxLayer_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);  
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'depthEstRegLoss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'depthReg';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 1, 1));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 1, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
end
rmLayerName = 'obj_Recurrent1';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule1';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, netMat.layers(netMat.getLayerIndex(sName)).block.size(end), netMat.layers(netMat.getLayerIndex(sName)).block.size(end)));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', netMat.layers(netMat.getLayerIndex(sName)).block.size(end), ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayerAtRecurrent1');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule1_depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule1_depth';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, depthBinNum, depthBinNum));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', depthBinNum, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayerAtRecurrent1_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule1_depthEstRegLoss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule1_depthReg';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 1, 1));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 1, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
%     layerTop = sprintf('Recurrent1_depthReg');
%     netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
%     netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'obj_Recurrent2';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule2';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, netMat.layers(netMat.getLayerIndex(sName)).block.size(end), netMat.layers(netMat.getLayerIndex(sName)).block.size(end)));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', netMat.layers(netMat.getLayerIndex(sName)).block.size(end), ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayerAtRecurrent2');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule2_depthEstSoftmaxloss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule2_depth';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, depthBinNum, depthBinNum));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', depthBinNum, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayerAtRecurrent2_depth');
    netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end
rmLayerName = 'recurrentModule2_depthEstRegLoss';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    
    baseName = 'recurrentModule2_depthReg';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 1, 1));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 1, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;
    
%     layerTop = sprintf('Recurrent1_depthReg');
%     netMat.addLayer(layerTop, dagnn.SoftMax(), sName, layerTop);
%     netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

outputIdx = netMat.layers(netMat.getLayerIndex('recurrentModule2_depthReg')).outputIndexes;
netMat.vars(outputIdx).precious = 1;
outputIdx = netMat.layers(netMat.getLayerIndex('recurrentModule1_depthReg')).outputIndexes;
netMat.vars(outputIdx).precious = 1;
outputIdx = netMat.layers(netMat.getLayerIndex('depthReg')).outputIndexes;
netMat.vars(outputIdx).precious = 1;
