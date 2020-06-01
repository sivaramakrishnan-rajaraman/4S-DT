
%% load images
CXR_dataset='...........................';
CXR_dataset= imageDatastore(CXR_dataset,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readAndPreprocessImage);

% determine the smallest amount of images in a category
tbl = countEachLabel(CXR_dataset)


%% Load a pretrained GoogLeNet network
net=resnet18; 

% convert the list of layers in net.Layers into a layer graph.
lgraph = layerGraph(net);

%% Replace the last learnable layer(fully connected layer) and the final classification layer with new layers adapted to the new data set.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
% The new fully connected layer
newLearnableLayer =fullyConnectedLayer(4,'Name','new_FC','WeightL2Factor',1,'WeightLearnRateFactor',10,'BiasLearnRateFactor',20);
newLearnableLayer.Weights= randn([4 512]) * 0.0001;
newLearnableLayer.Bias= randn([4 1])*0.0001 + 1; 
newLearnableLayer.WeightLearnRateFactor=10;
newLearnableLayer.BiasLearnRateFactor=20;

% The new classification layer
newClassLayer =classificationLayer('Name','new_classoutput');

% Replace these new layers within the layers lgraph
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% conncet these new layers within the layers lgraph
layers = lgraph.Layers;
connections = lgraph.Connections;
lgraph = createLgraphUsingConnections(layers,connections);

%% 
% Shuffle files in ImageDatastore
CXR_dataset = shuffle(CXR_dataset);

%% devide the dataset into 2 groups: 70% for trainingset and 30% for testset
[imdsTrainingSet,imdsTestSet]=splitEachLabel(CXR_dataset,0.7,'randomize');


%%
% hyper parameters for training the network
maxEpochs = 256;
miniBatchSize = 256;
numObservations = numel(imdsTrainingSet.Files);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

opts = trainingOptions('sgdm',...
                    'Initiallearnrate',0.0001,...
                    'Minibatchsize',miniBatchSize,...   
                    'maxEpoch',maxEpochs,'Shuffle','every-epoch',...            
                    'L2Regularization',0.001,'Momentum',0.9,...
                    'Plots','training-progress','LearnRateSchedule', 'piecewise', ...    
                    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.9,'LearnRateDropPeriod',2,...
                    'CheckpointPath' ,'E:\.............................');
                
%% Train the network using the training set using GPU Hardware 
[trainedNet,traininfo] = trainNetwork(imdsTrainingSet,lgraph,opts);

%% load mat.file from CheckpointPath
load('E:\...........\net_checkpoint.mat');
layerCheckPoint = net.Layers;
connectionCheckPoint = net.Connections;
lgraphCheckpoint = createLgraphUsingConnections(layerCheckPoint,connectionCheckPoint);

[predictedlabels,scores] = classify(netCeckPoint,imdsTestSet);  
accuracy = mean(predictedlabels == imdsTestSet.Labels);

save('Learned_features','trainedNet');
