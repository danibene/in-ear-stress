%% set paths

% see if a folder called "data" is in the current directory
% if not keep going to the parent directory
% until the main directory containing all of the code and data is found
baseDirPath = pwd;
while ~isfolder(fullfile(baseDirPath,'data'))
    [baseDirPath,~,~] = fileparts(baseDirPath);
end
% add the code in the entire directory
addpath(genpath(baseDirPath))

% load the filepaths for existing data
paths = loadpaths(baseDirPath);
%%

allSuccDiff = [];
allSuccDiffTime = [];
pIdxPerSuccDiff = [];
for pIdx=1:31
    [ibi,ibiTime] = readecg(paths,pIdx,'times',[0,Inf]);
    [succDiff,succDiffTime] = findsuccdiff(ibi,ibiTime); 
    pIdxPerSuccDiff = [pIdxPerSuccDiff;pIdx*ones(size(succDiff))];
    allSuccDiffTime = [allSuccDiffTime; succDiffTime];
    allSuccDiff = [allSuccDiff;succDiff];
end
%%
figure
boxplot(allSuccDiff)
%%
[M,I] = maxk(abs(allSuccDiff),10);
currSubs = pIdxPerSuccDiff(I);
currTimes = allSuccDiffTime(I);
toCorrTimes = NaN(size(currSubs));
toAddTimes = NaN(size(currSubs));
evaluated = zeros(size(currSubs));
t = array2table([currSubs,currTimes,toCorrTimes,toAddTimes,evaluated],'VariableNames',...
    {'relSubs','relTimes','toCorrTimes','toAddTimes','evaluated'});
dirPath = fullfile(paths.derivativesDirPath,"ecg_find_errors");
if ~isfolder(dirPath)
    mkdir(dirPath)
end
filePath = fullfile(dirPath,'ecg_time_corrections_todo.xls');
writetable(t,filePath)
%%
allSuccDiff = [];
allSuccDiffTime = [];
pIdxPerSuccDiff = [];
for pIdx=1:31
    [ibi,ibiTime] = readecg(paths,pIdx,'timeRef','ecg');
    delay = finddelayecg2arp(paths,pIdx);
    ibi = ibi(ibiTime >= delay);
    ibiTime = ibiTime(ibiTime >= delay);
    [succDiff,succDiffTime] = findsuccdiff(ibi,ibiTime); 
    pIdxPerSuccDiff = [pIdxPerSuccDiff;pIdx*ones(size(succDiff))];
    allSuccDiffTime = [allSuccDiffTime; succDiffTime];
    allSuccDiff = [allSuccDiff;succDiff];
end
%%
figure
boxplot(allSuccDiff)
%%
[M,I] = maxk(abs(allSuccDiff),10);

I = isoutlier(allSuccDiff,'percentiles',[0.1,99.9]);
ol = allSuccDiff(I);

currSubs = pIdxPerSuccDiff(I);
currTimes = allSuccDiffTime(I);
toCorrTimes = NaN(size(currSubs));
toAddTimes = NaN(size(currSubs));
unreliableTimes = NaN(size(currSubs));
evaluated = zeros(size(currSubs));
t = array2table([currSubs,currTimes,toCorrTimes,toAddTimes,unreliableTimes,evaluated],'VariableNames',...
    {'currSubs','currTimes','toRemTimes','toAddTimes','unreliableTimes','evaluated'});
dirPath = fullfile(paths.derivativesDirPath,"ecg_find_errors");
if ~isfolder(dirPath)
    mkdir(dirPath)
end
% filePath = fullfile(dirPath,'ecg_time_corrections.xls');
% writetable(t,filePath)
%%
for outlierIdx=1:length(currTimes)
    relSub = currSubs(outlierIdx);
    relTime = currTimes(outlierIdx);
    [ibiData,ibiTime] = readecg(paths,relSub);
    [rawDataCh1,rawTime] = readecg(paths,relSub,'output','raw','chanNum',1);
    [rawDataCh2,~] = readecg(paths,relSub,'output','raw','chanNum',2);
    ecgFs = 1/median(diff(rawTime));

    %%
    figure('units','normalized','outerposition',[0 0 1 1])
    xlimSubPlots = [relTime-10,relTime+10];
    ax1 = nexttile;
    plot(ibiTime,60./(ibiData./1000))
    hold on
    scatter(relTime,60/(ibiData(ibiTime==relTime)/1000),'LineWidth',3)
    xlim(xlimSubPlots)
    ax2 = nexttile;
    plot(rawTime,rawDataCh1)
    hold on
    scatter(ibiTime,rawDataCh1(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
    hold on
    scatter(relTime,rawDataCh1(round((relTime - ibiTime(1) + 1).*ecgFs)),'LineWidth',3)
    xlim(xlimSubPlots)
    ax3 = nexttile;
    plot(rawTime,rawDataCh2)
    hold on
    scatter(ibiTime,rawDataCh2(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
    hold on
    scatter(relTime,rawDataCh2(round((relTime - ibiTime(1) + 1).*ecgFs)),'LineWidth',3)
    xlim(xlimSubPlots)
    
    sgtitle(strcat("Participant ",num2str(relSub)))
    userInput = input(['If this not is a technical artifact, type NaN' ...
        '\nOtherwise, if the beat time should change, type the time of the beat that needs to be change (1 decimal point should do)' ...
        '\nIf a beat should be added, type Inf, if deleted, -Inf']);
    toCorrTimes = [toCorrTimes, userInput];

    if userInput>0
        userInput = input(['At what time should a beat be added?']);
        toAddTimes(userInput);
    end
    
    close
end
%% 
matData = readecg(paths,10,'output','mat');
[rawDataCh1,rawTime] = readecg(paths,10,'output','raw','chanNum',1);
[rawDataCh2,~] = readecg(paths,10,'output','raw','chanNum',2);

[ibiData,ibiTime] = readecg(paths,10);
ecgFs = 1/median(diff(rawTime));

figure
ax1 = nexttile;
plot(ibiTime,60./(ibiData./1000))
ax2 = nexttile;
plot(rawTime,rawDataCh1)
hold on
scatter(ibiTime,rawDataCh1(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
ax3 = nexttile;
plot(rawTime,rawDataCh2)
hold on
scatter(ibiTime,rawDataCh2(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
%%
[~,rawTime] = readecg(paths,10,'output','raw','chanNum',1,'timeRef','ecg');
[~,ibiTime] = readecg(paths,10,'timeRef','ecg');

figure
ax1 = nexttile;
plot(ibiTime,60./(ibiData./1000))
ax2 = nexttile;
plot(rawTime,rawDataCh1)
hold on
scatter(ibiTime,rawDataCh1(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
ax3 = nexttile;
plot(rawTime,rawDataCh2)
hold on
scatter(ibiTime,rawDataCh2(round((ibiTime - ibiTime(1) + 1).*ecgFs)))
%%
garde=find(matData.Mkr.vRejectU_RR==0 & matData.Mkr.vVSV==0);

origIbiTime=matData.Mkr.t_Rcm(garde)/ecgFs; 
% corresponding interbeat intervals
origIbi=matData.Mkr.vRR(garde); 
%%
ecgAddedIbiData = matData.ecgAddedIbiData;
ibiTimeCombined = origIbiTime;
ibiCombined = origIbi;
for i = 1:length(ecgAddedIbiData.add)
    ibiTimeCombined = [ibiTimeCombined,ecgAddedIbiData.add(i).tacq];
    ibiCombined = [ibiCombined,ecgAddedIbiData.add(i).RR];
end   
[ibiTimeSorted,sortIdx] = sort(ibiTimeCombined);
ibiSorted = ibiCombined(sortIdx);
[~,ia,~] = unique(ibiTimeSorted,'stable');
allEcgIbiTime = ibiTimeSorted(ia);
allEcgIbi = ibiSorted(ia);
%%
addIbiTime = [];
addIbi = [];
for i = 1:length(ecgAddedIbiData.add)
    addIbiTime = [addIbiTime,ecgAddedIbiData.add(i).tacq];
    addIbi = [addIbi, ecgAddedIbiData.add(i).RR];
end   
figure
plot(origIbiTime,60./(origIbi./1000))
hold on
plot(allEcgIbiTime,60./(allEcgIbi./1000))
hold on
%scatter(ecgAddedIbiData.add.tacq,60./(ecgAddedIbiData.add.RR.')/1000))
%%
figure
plot(ibiTimeCombined)
hold on
plot(ibiTimeSorted)
hold on
plot(allEcgIbiTime)
%%
function [succDiff,succDiffTime] = findsuccdiff(ibi,ibiTime)
    x = ibi(1:end-1);
    y = ibi(2:end);
    shouldIbi = diff(ibiTime)*1000;
    acceptDiffTrueShouldIbi = 0.000001;
    shouldIbiTime = ibiTime(2:end);
    discontinuityIdx = (abs(y-shouldIbi) > ...
        acceptDiffTrueShouldIbi);
    discontinuityTime = shouldIbiTime(discontinuityIdx);
    yTime = ibiTime(2:end);
    yOnlySucc = y;
    yOnlySucc(ismember(yTime,discontinuityTime)) = NaN;
    
    succDiffWithNan = yOnlySucc - x;
    succDiff = succDiffWithNan(~isnan(succDiffWithNan));
    succDiffTime = yTime(~ismember(yTime,discontinuityTime));
end