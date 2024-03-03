clc; clear; close all; warning off all;

image_folder = 'Training';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);

%Menginisialisasi matriks untuk menyimpan fitur dari data training
data_latih = zeros(10, total_images);

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
    
    CiriMEAN = mean(stats.Contrast);
    CiriENT = entropy(glcm);
    CiriVAR = var(stats.Correlation);

    % Min-Max Scaling
    minmax_scaled_data = mapminmax([CiriR; CiriG; CiriB; mean(area); mean(perimeter); mean(eccentricity); mean(metric); mean(stats.Contrast); entropy(glcm(:)); var(stats.Correlation)], 0, 1);

    % Store the features in the data matrix
    data_latih(:, n) = minmax_scaled_data
end

% Pembentukan target uji
data_latih = data_latih;
target_latih = zeros(1, total_images);
target_latih(:, 1:16) = 1;
target_latih(:, 17:32) = 2;
target_latih(:, 33:48) = 3;
target_latih(:, 47:64) = 4;
target_latih(:, 65:total_images) = 5;

% performance goal (MSE)
error_goal = 1e-6;

% choose a spread constant
spread = 30;

% choose max number of neurons
K = 25;

% number of neurons to add between displays
Ki = 10;

try
    % create a neural network
    net = newrb(data_latih, target_latih, error_goal, spread, K, Ki);

    % Optimize further
    net.trainFcn = 'trainlm';
    net.trainParam.epochs = 1500;  % Adjust the number of epochs

    % Proses training
    [net_keluaran, tr, ~, E] = train(net, data_latih, target_latih);

    save net_keluaran net_keluaran

    % Hasil Training
    hasil_latih = round(sim(net_keluaran, data_latih));
    correct_predictions = sum(hasil_latih == target_latih, 'all');
    akurasi = correct_predictions / numel(target_latih) * 100;

    disp(['Akurasi: ' num2str(akurasi) '%']);

catch ME
    disp(['Error: ' ME.message]);
end