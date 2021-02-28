clear all
close all
clc

% Notes:
% 1- Train using higher density of SnP e.g. 0.3 and then test on either higher or lower. 
% 2- Tune SVM box constraint parameter.


img_dir = 'D:\RnD\Frameworks\Dataset\UCS_SIPI\misc\';
img_list = dir(img_dir);
Options.mode = 0;       %0-Testing 1-Training
Options.disp = 1;
Options.noise = 0;      %0-snp, 1-gaussian, 2-both
Options.snpDensity = 0.1;
Options.gaussMean = 0;
Options.gaussVar = 20/(65025);
Options.filter = 7;     %0-Median,1-Gaussian,2-Wiener,3-Bilateral,4-SwitchingMedian, 5-NLM, 6-SVM_MED_BILAT, 7-SW_SVM_MED
Options.med_filt_kernel = [5,5];
Options.gaussFiltSigma = 0.85;
Options.wienerFiltKernel = [3,3];
Options.bilatFiltDoS = 1000;
Options.bilatFiltVar = 0.5;
Options.nlmFiltDoS = 5;
Options.mdlSVM = 'Mdl_SW_SVM_MED';%'Mdl_SVM_MED_BILAT';

len = length(img_list);
imgMetricsArr = zeros(len-2,6);
Xtrain = [];
Ytrain = [];
mdl = [];
if(Options.mode == 0)
    load(Options.mdlSVM,'mdl');
    Options.mdl = mdl;
end
        
for k=3:1:len
    k
    img = imread([img_dir,img_list(k).name]);
%     img = rgb2gray(img);
%     img = imresize(img, [256,256]);
    if(Options.noise == 0)
        imgNoisy = imnoise(img,'salt & pepper', Options.snpDensity);
    elseif(Options.noise == 1)
        imgNoisy = imnoise(img,'gaussian',Options.gaussMean,Options.gaussVar);
    elseif(Options.noise == 2)
        imgNoisy = imnoise(img,'salt & pepper', Options.snpDensity);
        imgNoisy = imnoise(imgNoisy,'gaussian',Options.gaussMean,Options.gaussVar);
    end
    
%     hist(double(imgNoisy(:)),256)% check noisy image stats
%     pause
    if(Options.mode == 0)   %Testing
        [imgMetrics,imgRec] = denoiser(img,imgNoisy,Options);
        imgMetricsArr(k,1) = imgMetrics.psnr_noisy;
        imgMetricsArr(k,2) = imgMetrics.psnr_rec;
%         imgMetrics.psnr_rec
        imgMetricsArr(k,3) = imgMetrics.snr_noisy;
        imgMetricsArr(k,4) = imgMetrics.snr_rec;
        imgMetricsArr(k,5) = imgMetrics.ssim_noisy;
        imgMetricsArr(k,6) = imgMetrics.ssim_rec;     

        if(Options.disp)
            close all
            imshow(img)
            figure
            imshow(imgNoisy)
            title(['Noisy SNR = ',num2str(imgMetrics.snr_noisy),' SSIM = ',num2str(imgMetrics.ssim_noisy)])
            figure
            imshow(imgRec)
            title(['Recovered SNR = ',num2str(imgMetrics.snr_rec),' SSIM = ',num2str(imgMetrics.ssim_rec)])
            pause
         end
    else %Training
        %Using rgb2gray will not work for SnP
        s = size(img);
        if(length(s)>2)
            img = rgb2gray(img);
            imgNoisy = rgb2gray(imgNoisy);
        end

        if(Options.filter == 6)
            [f1,f2,f3,f4,f5,f6,f7,f8,labels] = calcFeaturesLabels_SVM1(double(img),double(imgNoisy));
        end
        if(Options.filter == 7)
            [f1,f2,f3,f4,f5,f6,f7,f8,labels] = calcFeaturesLabels_SVM2(double(img),double(imgNoisy));
        end
        imgFeatures = [f1(:),f2(:),f3(:),f4(:),f5(:),f6(:),f7(:),f8(:)];
        Xtrain = [Xtrain;imgFeatures];
        Ytrain = [Ytrain;labels(:)];
%         
%         subplot(3,3,1); imshow(rgb2gray(imgNoisy))
%         subplot(3,3,2); imshow(uint8(abs(f1))); 
%         subplot(3,3,3); imshow(uint8(abs(f2))); 
%         subplot(3,3,4); imshow(uint8(abs(f3))); 
%         subplot(3,3,5); imshow(uint8(abs(f4))); 
%         subplot(3,3,6); imshow(uint8(abs(f5))); 
%         subplot(3,3,7); imshow(uint8(abs(f6))); 
%         subplot(3,3,8); imshow(uint8(abs(f7))); 
%         subplot(3,3,9); imshow(uint8(abs(f8))); 
%         pause
    end
end

if(Options.mode == 0)   %Testing
    ssim_rec_avg = mean(imgMetricsArr(:,6))
    snr_rec_avg = mean(imgMetricsArr(:,4))
    psnr_rec_avg = mean(imgMetricsArr(:,2))
%     ssim_avg = 0;
%     ssimArr_77 = [];
%     for k=1:length(img_list)-2
%         ssim_avg = ssim_avg + imgMetricsArr(k).ssim_rec;
%         ssimArr_77 = [imgMetricsArr(k).ssim_rec,ssimArr_77];
%     end
%     ssim_avg = ssim_avg/(length(img_list)-2)
%     save ssimArr_77;
else                    %Training
    idy0 = find(Ytrain==0);
    idy1 = find(Ytrain==1);
    ns = min(min(length(idy0),length(idy1)),5000);
    X0 = Xtrain(idy0,:);
    Y0 = Ytrain(idy0,:);
    X1 = Xtrain(idy1,:);
    Y1 = Ytrain(idy1,:);
    
    [X00,idx0] = datasample(X0,ns);
    Y00 = Y0(idx0);
    [X11,idx1] = datasample(X1,ns);
    Y11 = Y1(idx1);
    X = [X00;X11];
    Y = [Y00;Y11];
%     pause
%     mdl = fitcsvm(X,Y,'KernelFunction','rbf', 'BoxConstraint',1,'ClassNames',[0,1]);
    mdl = fitcsvm(X,Y,'KernelFunction','linear', 'BoxConstraint',1,'ClassNames',[0,1]);
%     mdl = fitcsvm(X,Y,'KernelFunction','hik', 'BoxConstraint',10,'ClassNames',[1,2]);
%     YY=mdl.predict(X);
    YY=double((Xtrain*mdl.Beta + mdl.Bias)>0);
%     err = sum(abs(YYp-YY))
%     YY = Xtrain*[2.0727,0,0,0,0,0,0,0]-1;
    e = abs(YY-Ytrain);
    1-sum(e)/length(Ytrain)
%     keyboard
    delete([Options.mdlSVM,'.mat']);
    save(Options.mdlSVM,'mdl');
end

