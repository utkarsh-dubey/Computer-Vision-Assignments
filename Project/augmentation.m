function augmentation(data_dir)
%% data augmentation script for SK-LARGE
if nargin == 0
    skDir = './';
else
    skDir = data_dir;
end 
trnImgDir = fullfile(skDir,'images/train/');
trnGtDir = fullfile(skDir,'groundTruth/train/');
scales = [0.8,1.0,1.2];
orients = [0,90,180,270];
flips = [0,1,2];
augDir = 'aug_data';  % Directory to save the augmented data
if ~exist(augDir,'dir')
    mkdir(augDir);
end
pair_list = fopen(fullfile(augDir,'train_pair.lst'),'w');
imgs = dir(fullfile(trnImgDir,'*.jpg'));
imgs = {imgs.name}; nImgs = length(imgs);
for i=1:nImgs,imgs{i} = imgs{i}(1:end-4);end
for i=1:nImgs
   im = imread([trnImgDir,imgs{i},'.jpg']);
   gt = load([trnGtDir,imgs{i},'.mat']);
   edge = single(gt.edge);  symmetry = single(gt.symmetry);
   for s = scales
      edge1 = imresize(edge,s,'bilinear');
      symmetry1 = imresize(symmetry,s,'bilinear');
      im0 = imresize(im,s,'bilinear');
      scale = bwdist(edge1); 
      scale(scale<1 & symmetry1~=0)=1;
      scale(symmetry1 == 0) = 0; scale = scale*2;
      maxs = max(scale(:));
      if(maxs>=150)
        disp([imgs{i},'s:',num2str(s),'max scale:',num2str(maxs)]);
      end
      scale = uint8(scale);
      
      for o = orients
         im1 = imrotate(im0,o);
         scale1 = imrotate(scale,o);
         for f = flips
            switch f
                case 0
                    im2 = im1;
                    scale2 = scale1;
                case 1
                    im2 = fliplr(im1);
                    scale2 = fliplr(scale1);
                case 2
                    im2 = flipud(im1);
                    scale2 = flipud(scale1);
                otherwise
                    error('unknown flip type');
            end
             imSavDir = fullfile(augDir,'im_scale',num2str(s),'/o',num2str(o),'/f',num2str(f),'/');
             gtSavDir = fullfile(augDir,'gt_scale',num2str(s),'/o',num2str(o),'/f',num2str(f),'/');
             if(~exist(imSavDir,'dir')),mkdir(imSavDir);end
             if(~exist(gtSavDir,'dir')),mkdir(gtSavDir);end
             imSiz = size(im2);imSiz = imSiz(1:2);
             gtSiz = size(scale2);
             assert(nnz(imSiz-gtSiz)==0);
             imwrite(im2,[imSavDir,imgs{i},'.jpg']);
             imwrite(scale2,[gtSavDir,imgs{i},'.png']);
             fprintf(pair_list,[imSavDir,imgs{i},'.jpg',' ',gtSavDir,imgs{i},'.png\r\n']);
         end
      end
   end
   disp([num2str(i),' of ',num2str(nImgs)]);
end
end  % end of the function

