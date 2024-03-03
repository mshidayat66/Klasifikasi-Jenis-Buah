image_folder = 'Test';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);

data_uji = zeros(10, total_images);

for n = 1:total_images
    full_name = fullfile(image_folder, filenames(n).name);
    Img = imread(full_name);

    %Segmentasi Citra
    grayImage = rgb2gray(Img);
    threshold = graythresh(grayImage);
    binaryImage = im2bw(grayImage, threshold);
    se = strel('disk', 5);
    closedImage = imclose(binaryImage, se);
    segmentedImage = Img;
    segmentedImage(repmat(closedImage, [1, 1, 3])) = 255;
    
    %Ekstraksi Ciri Warna RGB
    R = segmentedImage(:,:,1);
    G = segmentedImage(:,:,2);
    B = segmentedImage(:,:,3);
    
    CiriR = mean2(R);
    CiriG = mean2(G);
    CiriB = mean2(B);
    
    %Ekstraksi Ciri Ukuran Buah
    J = rgb2hsv(segmentedImage);
    
    H = J(:,:,1);
    S = J(:,:,2);
    V = J(:,:,3);
    
    bw = im2bw(S,.25);
    
    bw = imfill(bw,'holes');
    bw = bwareaopen(bw,100);
    
    [bw,num] = bwlabel(bw);
    bbox = zeros(num,4);
    
    for k = 1:num
        bw2 = bw==k;
        [B,L] = bwboundaries(bw2,'noholes');
        stats = regionprops(L,'All');
        perimeter = cat(1,stats.Perimeter);
        area = cat(1,stats.Area);
        eccentricity = cat(1, stats.Eccentricity);
        metric = 4*pi*area/perimeter^2;
        bbox(k,:) = cat(1,stats.BoundingBox);
    end
    % Ekstraksi Ciri Tekstur menggunakan GLCM
    grayImage = rgb2gray(segmentedImage);
    
    % Hitung GLCM
    offset = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(grayImage, 'Offset', offset);

    % Hitung statistik GLCM
    stats = graycoprops(glcm);
    
    % Min-Max Scaling
    minmax_scaled_data = mapminmax([CiriR; CiriG; CiriB; mean(area); mean(perimeter); mean(eccentricity); mean(metric); mean(stats.Contrast); entropy(glcm(:)); var(stats.Correlation)], 0, 1);

    % Store the features in the data matrix
    data_uji(:, n) = minmax_scaled_data;
end

% Pembentukan target uji
data_uji = data_uji;
target_uji = zeros(1, total_images);
target_uji(:, 1:4) = 1;
target_uji(:, 5:8) = 2;
target_uji(:, 9:12) = 3;
target_uji(:, 13:16) = 4;
target_uji(:, 17:20) = 5;

% Load the pre-trained network
load net_keluaran;

% Hasil Uji
hasil_uji = round(sim(net_keluaran, data_uji));
correct_predictions = sum(hasil_uji == target_uji, 'all');
akurasi = correct_predictions / numel(target_uji) * 100;

% Display accuracy
disp(['Akurasi: ' num2str(akurasi) '%']);
% Display Confusion Matrix as a Chart for Testing
confMat_uji = confusionmat(target_uji, hasil_uji);
figure;
confusionchart(confMat_uji, {'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'}, 'Title', 'Confusion Matrix for Testing');