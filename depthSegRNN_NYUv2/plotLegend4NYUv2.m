function plotLegend4NYUv2(NYUv2_label2color)
%% prepare legend
name2color = NYUv2_label2color.class_names(1:end-1);
legendH = 50;
legendW = 25;
legendMat = NYUv2_label2color.mask_cmap(2:numel(name2color)+1,:);
legendMat = reshape(legendMat, [1, 1, numel(name2color), 3]);
legendMat = repmat(legendMat, [legendH, legendW, 1, 1]);
legendMat = reshape(legendMat, [legendH, legendW*numel(name2color), 3]);
legendMat = imrotate(legendMat, -90);
legendMat = cat(2, legendMat, ones(size(legendMat,1), 200, 3));

legendMat = cat(2, legendMat(1:size(legendMat,1)/4,:,:), ...
    legendMat(size(legendMat,1)/4+1:size(legendMat,1)/4*2,:,:), ...
    legendMat(size(legendMat,1)/4*2+1:size(legendMat,1)/4*3,:,:), ...
    legendMat(size(legendMat,1)/4*3+1:size(legendMat,1)/4*4,:,:) );

imagesc(legendMat); axis image off;
for i = 1:10
    text(legendH+7, (i-1)*legendW+10, name2color{i}, 'Interpreter', 'none' );
end
for i = 11:20
    text(legendH+7 + 250, (i-11)*legendW+10, name2color{i}, 'Interpreter', 'none' );
end
for i = 21:30
    text(legendH+7 + 500, (i-21)*legendW+10, name2color{i}, 'Interpreter', 'none' );
end
for i = 31:numel(name2color)
    text(legendH+7 + 750, (i-31)*legendW+10, name2color{i}, 'Interpreter', 'none' );
end

