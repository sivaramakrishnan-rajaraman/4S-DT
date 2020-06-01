
%% load images
NewDataset='E:\..................................';
NewDataset= imageDatastore(NewDataset,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readAndPreprocessImage);

% count the number of classes withing the NewDataset
% dir('dirPath')
OutputClassese= length(dir('E:\..........'))-2 ;
countEachLabel(NewDataset)

%% Load self_Supervised ResNet for transfer learning
load('Learned_features.mat')

layers = trainedNet.Layers;
connections = trainedNet.Connections;
lgraph = createLgraphUsingConnections(layers,connections);

%% Replace the last learnable layer(fully connected layer) and the final classification layer with new layers adapted to the new data set.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
% The new fully connected layer
newLearnableLayer =fullyConnectedLayer(OutputClassese,'Name','new_FC','WeightL2Factor',1);
newLearnableLayer.Weights= randn([OutputClassese 512]) * 0.0001;
newLearnableLayer.Bias= randn([OutputClassese 1])*0.0001 + 1; 
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
NewDataset = shuffle(NewDataset);

%% devide the dataset into 2 groups: 70% for trainingset and 30% for testset
[imdsTrainingSet,imdsTestSet]=splitEachLabel(NewDataset,0.7,'randomize');

%% Hyper-parameters
maxEpochs = 100;
miniBatchSize = 256;
numObservations = numel(trainingimages.Files);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);


opts = trainingOptions('sgdm',...
                    'Initiallearnrate',0.01,...
                    'Minibatchsize',miniBatchSize,...   
                    'maxEpoch',maxEpochs,'Shuffle','every-epoch',...            
                    'L2Regularization',0.001,'Momentum',0.9,...
                    'Plots','training-progress',...
                    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor',...
                    0.9,'LearnRateDropPeriod',2);
    

% Training the Self supervised model for the new task
[trainedNet,traininfo] = trainNetwork(imdsTrainingSet,lgraph,opts);

[predictedlabels,scores] = classify(trainedNet,imdsTestSet);  
                 
%% compute the confusion matrix 
[cmat,classNames] = confusionmat(imdsTestSet.Labels, predictedlabels); 
 cm = confusionchart(cmat,classNames);

% Sort Classes
sortClasses(cm,["Covid_19","SARS","normal"])
cmat=cm.NormalizedValues;
cmat=confusionchart(cmat,classNames);
          
[acc, sn, sp]= ConfusionMat_MultiClass (cmat,classNames);
classification_Accuaracy    = acc;
classification_sensitivity  = sn;
classification_specifity    = sp;

 %% ********************* plot ROc curve *****************
  
targets=grp2idx(imdsTestSet.Labels);

[X,Y,Threshold,AUCpr] = perfcurve(targets, scores(:,1), 1, 'xCrit', 'fpr', 'yCrit', 'tpr');
plot(X,Y)
xlabel('1-specificity'); ylabel('sensitivity');xlim([0 0.4]);ylim([0.5 1]);
title(['ROC analysis for 4S_DT (AUC: ' num2str(AUCpr) ')'])

