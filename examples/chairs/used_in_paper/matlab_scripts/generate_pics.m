net_path = 'FOLDER_WHERE_THE_NET_IS';
output_path = 'FOLDER_WHERE_TO_PUT_GENERATED_PICTURES';


%% teaser morphing
params = struct;
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

label_in = randi(843, 1, 1, 1, batch_size, 'single') - 1;
onehot_in = zeros(1, 1, 843, batch_size, 'single');
aug_params_in = zeros(1, 1, 28, batch_size, 'single');
theta = linspace(20,20,batch_size);
phi = linspace(301,301,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);

onehot_in(1, 1, 6, :) = linspace(1, 0,batch_size);
onehot_in(1, 1, 39, :) = linspace(0, 1,batch_size);

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {onehot_in; angles_in; aug_params_in});

nn_maps{2}(nn_maps{2} >= 1) = 1;
nn_maps{2}(nn_maps{2} < 0) = 0;
toshow = nn_maps{1} .* nn_maps{2} + ones(size(nn_maps{1})) .* (1 - nn_maps{2});
toshow = permute(single(toshow(:,:,:,[1 18 27 30 64 45 40 35])), [2 1 3 4]); 
pic = patchShow(toshow,'Color', 'cols', 4,'noshow');
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'CVPR15_teaser_morphing.png'));


%% augmentation
params = struct;
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

label_in = randi(843, 1, 1, 1, batch_size, 'single') - 1;
onehot_in = zeros(1, 1, 843, batch_size, 'single');
aug_params_in = zeros(1, 1, 28, batch_size, 'single');
theta = linspace(20,20,batch_size);
phi = linspace(301,301,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);

num_subsets = 7;
subset_size = 7;
selected_chairs = [374, 72, 49, 668, 290, 600, 27];
subset = {};
for ns = 1:num_subsets
    subset{ns} = [(ns-1)*subset_size+1: ns*subset_size];
    onehot_in(1,1,selected_chairs(ns),subset{ns}) = 1;
end

aug_params_in(1,1,2,subset{1}) = linspace(-0.2, 0.2, subset_size); % dx
aug_params_in(1,1,3,subset{1}) = linspace(-0.2, 0.2, subset_size); % dy
aug_params_in(1,1,4,subset{2}) = linspace(-0.5, 0.5, subset_size); % angle
aug_params_in(1,1,5,subset{3}) = linspace(log(1.), log(1.8), subset_size); % zoom_x
aug_params_in(1,1,6,subset{3}) = linspace(log(1.), log(1.8), subset_size); % zoom_y
aug_params_in(1,1,5,subset{4}) = linspace(log(1.5), -log(1.5), subset_size); % zoom_x
aug_params_in(1,1,6,subset{4}) = linspace(-log(1.5), log(1.5), subset_size); % zoom_y
aug_params_in(1,1,23,subset{5}) = linspace(-5., 5., subset_size); % sat_mult
aug_params_in(1,1,24,subset{5}) = aug_params_in(1,1,23,subset{5}); % sat_mult
aug_params_in(1,1,27,subset{6}) = linspace(-2., 2., subset_size); % lmult_mult
aug_params_in(1,1,28,subset{7}) = linspace(-3, 3, subset_size); % col_angle

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {onehot_in; angles_in; aug_params_in});

nn_maps{2}(nn_maps{2} >= 1) = 1;
nn_maps{2}(nn_maps{2} < 0) = 0;
toshow = nn_maps{1} .* nn_maps{2} + ones(size(nn_maps{1})) .* (1 - nn_maps{2});
toshow = permute(single(toshow(:,:,:,1:num_subsets*subset_size)), [2 1 3 4]); 
pic = patchShow(toshow,'Color', 'cols', subset_size,'noshow');
canvas_color = [0.8 0 0];
pic(:,129*3:129*3+2,:) = repmat(reshape(canvas_color,[1 1 3]), [size(pic,1), 3, 1]); 
pic(:,129*4-1:129*4+1,:) = repmat(reshape(canvas_color,[1 1 3]), [size(pic,1), 3, 1]); 
pic(1:3,129*3:129*4,:) = repmat(reshape(canvas_color,[1 1 3]), [3, 130, 1]); 
pic(end-2:end,129*3:129*4,:) = repmat(reshape(canvas_color,[1 1 3]), [3, 130, 1]);
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'augmentation.png'));

%% morphing
params = struct;
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

label_in = randi(843, 1, 1, 1, batch_size, 'single') - 1;
onehot_in = zeros(1, 1, 843, batch_size, 'single');
aug_params_in = zeros(1, 1, 28, batch_size, 'single');
theta = linspace(20,20,batch_size);
phi = linspace(301,301,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);

selected_pairs = {[776 11], [5 168], [742 572], [671  84], [786 739], [117 300], [624 537], [320 129]};
sample_points = [1 0.65 0.57 0.5 0.43 0.35 0];
subset_size = numel(sample_points);

subset = {};
for ns = 1:numel(selected_pairs)
    subset{ns} = [(ns-1)*subset_size+1: ns*subset_size];
    onehot_in(1,1,selected_pairs{ns}(1),subset{ns}) = sample_points;
    onehot_in(1,1,selected_pairs{ns}(2),subset{ns}) = 1-sample_points;
end

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {onehot_in; angles_in; aug_params_in});

nn_maps{2}(nn_maps{2} >= 1) = 1;
nn_maps{2}(nn_maps{2} < 0) = 0;
toshow = nn_maps{1} .* nn_maps{2} + ones(size(nn_maps{1})) .* (1 - nn_maps{2});
toshow = permute(single(toshow(:,:,:,1:numel(selected_pairs)*subset_size)), [2 1 3 4]); 
pic = patchShow(toshow,'Color', 'cols', subset_size,'noshow');
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'morphing.png'));

%% 1st layer filters
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
layers = caffe('get_weights');

toshow = single(layers(18).weights{1}); 
pic = patchShow(toshow,'Color', 'cols', 32, 'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'first_layer_filters_segm.png');
pause

toshow = single(layers(13).weights{1}); 
pic = patchShow(toshow,'Color', 'cols', 32,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'first_layer_filters_RGB.png'));

%% activate single neurons

params = struct;
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

onehot_in = zeros(1, 1, 843, batch_size, 'single');
aug_params_in = zeros(1, 1, 28, batch_size, 'single');

theta = linspace(30,30,batch_size);
phi = linspace(310,310,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);


for ni = 1:batch_size
    distr = ones(3,1);
%     distr = distr / sum(distr);
    onehot_in(1, 1, randperm(size(onehot_in,3),numel(distr)), ni) = distr;
%     onehot_in(1, 1, 5, ni) = ni/16;
end

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {onehot_in; angles_in; aug_params_in});

toshow = single(permute(nn_maps{1}, [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'cols', 8,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;

%% fc1
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

aug_params_in = zeros(1, 1, 28, batch_size, 'single');
fc1_label_in = zeros(1, 1, 512, batch_size, 'single');

theta = linspace(30,30,batch_size);
phi = linspace(310,310,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);

selected_neurons = [2 2 3 6 12 24 25];
for ni = 1:numel(selected_neurons)
    fc1_label_in(1, 1, selected_neurons(ni), ni) = 10;
end
fc1_label_in(1, 1, :, 1) = 0;

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc1_label_in; angles_in; aug_params_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_neurons)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_fc1.png'));

%% fc2
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

aug_params_in = zeros(1, 1, 28, batch_size, 'single');
fc2_label_in = zeros(1, 1, 512, batch_size, 'single');

theta = linspace(30,30,batch_size);
phi = linspace(310,310,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);

selected_neurons = [3 3 6 16 17 19 32];
for ni = 1:numel(selected_neurons)
    fc2_label_in(1, 1, selected_neurons(ni), ni) = 15;
end
fc2_label_in(1, 1, :, 1) = 0;

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc2_label_in; angles_in; aug_params_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_neurons)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_fc2.png'));

%% fc3
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc3_in = zeros(1, 1, 1024, batch_size, 'single');

selected_neurons = [2 28 4 17 40 26 41];
for ni = 1:numel(selected_neurons)
    fc3_in(1, 1, selected_neurons(ni), ni) = 5;
end
fc3_in(1, 1, :, 1) = 0;

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc3_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_neurons)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_fc3.png'));

%% fc4
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc4_in = zeros(1, 1, 1024, batch_size, 'single');

selected_neurons = [4 4 6 28 20 29 49];
for ni = 1:numel(selected_neurons)
    fc4_in(1, 1, selected_neurons(ni), ni) = 8;
%     fc4_in(1, 1, floor((ni+7)/8), ni) = 2*(mod(ni-1,8) + 1);
end
fc4_in(1, 1, :, 1) = 0;

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc4_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_neurons)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_fc4.png'));

%% deconv7
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

deconv7_in = zeros(32, 32, 92, batch_size, 'single');

selected_filters = sort([1 6 7 8 10 12 14 27 33 51 48 64 57 60 45 2]);
% selected_filters = 1:batch_size;
for ni = 1:numel(selected_filters)
    deconv7_in(16, 16, selected_filters(ni), ni) = 1;
%     deconv7_in(16, 16, randperm(size(deconv7_in,3),20), ni) = 10;
end

% deconv7_in(16, 16, 1, :) = linspace(0,1, batch_size);
% deconv7_in(16, 16, 60, :) = linspace(1,0, batch_size);

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {deconv7_in});

toshow = single(permute(nn_maps{1}(55:66,55:66,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'cols', 16,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_deconv7.png'));

%% deconv6
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

deconv6_in = zeros(16, 16, 256, batch_size, 'single');

selected_filters = sort([2 3 5 6 14 18 35 38]);
for ni = 1:numel(selected_filters)
    deconv6_in(8, 8, selected_filters(ni), ni) = 10;
%     deconv6_in(8, 8, randperm(size(deconv6_in,3),10), ni) = 10;
end

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {deconv6_in});

toshow = single(permute(nn_maps{1}(44:71,44:71,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'cols', 8,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_deconv6.png'));

%% fc5_reshape
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc5_reshape_in = zeros(8, 8, 256, batch_size, 'single');

selected_filters = sort([2 3 4 44]);
% selected_filters = 1:batch_size;
for ni = 1:numel(selected_filters)
    fc5_reshape_in(4, 4, selected_filters(ni), ni) = 0.1;
end

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc5_reshape_in});

% toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_filters)), [2 1 3 4])); 
toshow = single(permute(nn_maps{1}(24:80,24:80,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'cols', 4,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_fc5_reshape.png'));

%% reconstruct from real hidden activations
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

onehot_in = zeros(1, 1, 843, batch_size, 'single');
aug_params_in = zeros(1, 1, 28, batch_size, 'single');
% fc5_reshape_in = zeros(8, 8, 128, batch_size, 'single');
% fc5_segm_reshape_in = zeros(8, 8, 128, batch_size, 'single');
% fc5_reshape_in = zeros(16, 16, 128, batch_size, 'single');
% fc5_segm_reshape_in = zeros(16, 16, 92, batch_size, 'single');
theta = linspace(20,20,batch_size);
phi = linspace(301,301,batch_size);
angles_in = zeros(1, 1, 4, batch_size, 'single');
angles_in(1, 1, 1, :) = sin(theta/180*pi);
angles_in(1, 1, 2, :) = cos(theta/180*pi);
angles_in(1, 1, 3, :) = sin(phi/180*pi);
angles_in(1, 1, 4, :) = cos(phi/180*pi);
for ni = 1:batch_size
    onehot_in(1, 1, ni, ni) = 1;
end
% onehot_in(1, 1, 1, :) = linspace(0,1,batch_size);
% onehot_in(1, 1, 2, :) = linspace(1,0,batch_size);
caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_hidden_maps = caffe('forward', {onehot_in; angles_in; aug_params_in});

%% interesting last layer filter maps
% interesting_maps = [3 10 34 50 39];
interesting_maps = [3 10 34 50];

toshow = single(permute(nn_hidden_maps{14}(:,:,interesting_maps,3), [2 1 3 4])); 
pic = patchShow(toshow, 'rows', 1,'noshow','bgcolor', 0.);
figure(17);
imagesc(pic);
colormap gray
axis equal off;

imwrite(pic, fullfile(output_path, 'pattern_maps_deconv8.png');

toshow = single(permute(nn_hidden_maps{14}(20:40,20:40,interesting_maps,3), [2 1 3 4])); 
pic = patchShow(toshow, 'rows', 1,'noshow','bgcolor', 0.);
figure(17);
imagesc(pic);
colormap gray
axis equal off;

imwrite(pic, fullfile(output_path, 'pattern_maps_deconv8_closeup.png'));

%% deconv8 
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

deconv8_in = zeros(64, 64, 92, batch_size, 'single');

selected_chairs = [3 12];

deconv8_in(:, :, :, 1) =  nn_hidden_maps{14}(:, :, :, 3);
deconv8_in(:, :, :, 2) =  nn_hidden_maps{14}(:, :, :, 3);
% deconv8_in(:, :, [3 10 18 34 39 50], 2) = 0;
deconv8_in(:, :, [3 10 34 50], 2) = 0;
% deconv8_in(:, :, [1 2 4 5 6 7 8 9 10], 2) = 0;

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {deconv8_in});

toshow = zeros(128,128,3,4,'single');
toshow(:,:,:,1) = single(permute(nn_maps{1}(:,:,:,2), [2 1 3 4]));
toshow(:,:,:,2) = imresize(single(permute(nn_maps{1}(50:70,50:70,:,2), [2 1 3 4])), [128 128], 'nearest'); 
toshow(:,:,:,3) = single(permute(nn_maps{1}(:,:,:,1), [2 1 3 4]));
toshow(:,:,:,4) = imresize(single(permute(nn_maps{1}(50:70,50:70,:,1), [2 1 3 4])), [128 128], 'nearest'); 


pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure(17);
imagesc(pic);
axis equal off;
imwrite(pic, fullfile(output_path, 'pattern_artifacts.png'));

%% deconv6 
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

deconv6_in = zeros(16, 16, 256, batch_size, 'single');

selected_filters = sort([2 3 5 6 14 18 35 38]);
% selected_filters = 1:batch_size;
for ni = 1:numel(selected_filters)
    deconv6_in(8, 8, selected_filters(ni), ni) = 10;
%     deconv6_in(1:floor((ni-1)/4)+1, 1:floor((ni-1)/4)+1, :, ni) = nn_hidden_maps{12}(1:floor((ni-1)/4)+1, 1:floor((ni-1)/4)+1, :, ni);
%     deconv6_in(floor((ni-1)/8)*2+1:floor((ni-1)/8)*2+2, mod(ni-1,8)*2+1:mod(ni-1,8)*2+2, :, ni) =...
%         nn_hidden_maps{12}(floor((ni-1)/8)*2+1:floor((ni-1)/8)*2+2, mod(ni-1,8)*2+1:mod(ni-1,8)*2+2, :, ni);
%     deconv6_in(floor((ni-1)/8)*2+1:floor((ni-1)/8)*2+2, mod(ni-1,8)*2+1:mod(ni-1,8)*2+2, :, ni) =...
%         nn_hidden_maps{12}(floor((ni-1)/8)*2+1:floor((ni-1)/8)*2+2, mod(ni-1,8)*2+1:mod(ni-1,8)*2+2, :, ni);
end

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {deconv6_in});

% toshow = single(permute(nn_maps{1}(:,:,:,1:numel(selected_filters)), [2 1 3 4])); 
toshow = single(permute(nn_maps{1}(44:71,44:71,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'cols', 8,'noshow','bgcolor', -0.3);
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'generate_from_single_neurons_deconv6.png'));


%% fc5_reshape_vary_area
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');
batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc5_reshape_in = zeros(8, 8, 256, batch_size, 'single');

for ni = 1:4
    fc5_reshape_in(5-ni:ni+4, 5-ni:ni+4, :, ni) =  nn_hidden_maps{11}(5-ni:ni+4, 5-ni:ni+4, :, 10);
end
% fc5_reshape_in(:, :, 1:63, 64) =  nn_hidden_maps{11}(:, :, 1:63, 1);

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc5_reshape_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:4), [2 1 3 4])); 
% toshow = single(permute(nn_maps{1}(44:71,:,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 1,'noshow');
figure;
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'fc5_reshape_vary_area.png'));

%% fc5_reshape_checkerboard
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc5_reshape_in = zeros(8, 8, 256, batch_size, 'single');

for ni = 1:batch_size
%     fc5_reshape_in(1:2:end, 1:2:end, :, ni) =  nn_hidden_maps{11}(1:2:end, 1:2:end, :, ni);
%     fc5_reshape_in(2:2:end, 2:2:end, :, ni) =  nn_hidden_maps{11}(2:2:end, 2:2:end, :, ni);
    fc5_reshape_in(:, :, :, ni) =  nn_hidden_maps{11}(:, :, :, ni);
    fc5_reshape_in(2:7, 2:7, :, ni) =  0;
end
% fc5_reshape_in(:, :, 1:63, 64) =  nn_hidden_maps{11}(:, :, 1:63, 1);

caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc5_reshape_in});

toshow = single(permute(nn_maps{1}(:,:,:,:), [2 1 3 4])); 
% toshow = single(permute(nn_maps{1}(44:71,:,:,1:numel(selected_filters)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', 8,'noshow');
figure;
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'fc5_reshape_checkerboard.png'));

%% fc4 try neuron effects (N 28 - zoom neuron)
params.model_def_file = fullfile(net_path, 'generate.prototxt');
params.model_file = fullfile(net_path, 'trained_model');

batch_size = 64;

setenv('HDF5_DISABLE_VERSION_CHECK', '1');

fc4_in = zeros(1, 1, 1024, batch_size, 'single');

fc4_in = repmat(nn_hidden_maps{10}(:,:,:,15), [1 1 1 64]);

% selected_neurons = [1 4 6 28 20 29 49];
selected_neurons = [28];
for ni = 1:numel(selected_neurons)
    for ns = 1:7
        fc4_in(1, 1, selected_neurons(ni), (ni-1)*7 + ns) = fc4_in(1, 1, selected_neurons(ni), (ni-1)*7 + ns) + ns*5;
    end
end

caffe('init', params.model_def_file, params.model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
nn_maps = caffe('forward', {fc4_in});

toshow = single(permute(nn_maps{1}(:,:,:,1:7*numel(selected_neurons)), [2 1 3 4])); 
pic = patchShow(toshow,'Color', 'rows', numel(selected_neurons),'noshow');
figure(17);
imagesc(pic);
axis equal off;

imwrite(pic, fullfile(output_path, 'fc4_zoom_neuron.png'));









          

