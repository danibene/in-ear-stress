%% set paths

% see if a folder called "data" is in the current directory
% if not keep going to the parent directory
% until the main directory containing all of the code and data is found
% (change path if data is in a different directory)
baseDirPath = pwd;
while ~isfolder(fullfile(baseDirPath,'data'))
    [baseDirPath,~,~] = fileparts(baseDirPath);
end
% add the code in the entire directory 
% (change path if code is in a different directory)
addpath(genpath(baseDirPath))

% load the file paths
paths = loadpaths(baseDirPath,'saveFile','saveNewFile');
%%
subs = [1:26,28:31];
nTsks = 3;
diffIntendedRealAnticipation = NaN(length(subs),nTsks);
realTskTime = NaN(length(subs),nTsks);
count = 0;
for pIdx=subs
    count = count + 1;
    [v1,v2,diffIntendedRealAnticipation(count,:),realTskTime(count,:)] = ...
        loadtsktimes(paths,pIdx,'output','diffUniformOriginal');
end
%%
tskNames = ["MENTAL","NOISE","CPT"];

figure = tiledlayout(2,1);
ax1 = nexttile;
boxplot(diffIntendedRealAnticipation,tskNames)
title("Difference between intended anticipation and actual end of task preparation")

ax2 = nexttile;
boxplot(realTskTime,tskNames)
title("Actual task time")

