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
fileName = "ecg_time_corrections.xls";
fileDir = "ecg_find_erros";
filePath = fullfile(paths.derivativesDirPath,fileDir,fileName);
ecgErrorTable = readtable(filePath);
%%
subsInTable = unique(ecgtimecorrections1.currSubs);
for pIdx = subsInTable.'
    [ibiData,ibiTime] = readecg(paths,pIdx,'timeRef','ecg');
    subTable = ecgtimecorrections1(ecgtimecorrections1.currSubs == pIdx,:);
    for i=1:length(subTable.toAddTimes)
        if ~isnan(subTable.toAddTimes(i))
            [ibiTime, sortIdx] = sort([ibiTime;subTable.toAddTimes(i)]);
            [~,sIdx] = max(sortIdx);
            ibiData(sIdx+1:end+1) = ibiData(sIdx:end);
            %if ~isnan(ibiData(sIdx+1)) & ~isnan(ibiData(sIdx))
                ibiData(sIdx+1) = (ibiTime(sIdx+1)-ibiTime(sIdx))*1000;
            %end
            %if ~isnan(ibiData(sIdx-1)) & ~isnan(ibiData(sIdx))
                ibiData(sIdx) = (ibiTime(sIdx)-ibiTime(sIdx-1))*1000;
            %end
            %if ~isnan(ibiData(sIdx-1)) & ~isnan(ibiData(sIdx-2))
                ibiData(sIdx-1) = (ibiTime(sIdx-1)-ibiTime(sIdx-2))*1000;
            %end
        end
        % ibiData = [ibiData(1);diff(ibiTime)*1000];
        % ibiData = tableRow.toAddTimes;

    end
    for i=1:length(subTable.toRemTimes)
        if ~isnan(subTable.toRemTimes(i))
            [~,toRemIdx] = min(abs(ibiTime - subTable.toRemTimes(i)));
            ibiTime(toRemIdx) = [];
            if toRemIdx > 1
                ibiData(toRemIdx) = [];
                ibiData(toRemIdx) = (ibiTime(toRemIdx)-ibiTime(toRemIdx-1))*1000;
                ibiData(toRemIdx - 1) = (ibiTime(toRemIdx-1)-ibiTime(toRemIdx-2))*1000;
            else
                ibiData = ibiData(2:end);
            end
        end
    end
    afterChangesDB.ibi = ibiData;
    afterChangesDB.ibiTime = ibiTime;
    ecgAddedIbiFileName = strcat('add_',...
        paths.fileNameMatchTable.Mkr_ECG_name(pIdx),...
        '.mat');
    ecgAddedIbiFilePath = ...
        fullfile(paths.ecgAddedIbiDirPath,ecgAddedIbiFileName);
    if exist(ecgAddedIbiFilePath,'file')
        ecgAddedIbiData = load(cell2mat(ecgAddedIbiFilePath));
        if isfield(ecgAddedIbiData,'add')
            add = ecgAddedIbiData.add;
        end
        save(ecgAddedIbiFilePath,'add','afterChangesDB');
    else
        save(ecgAddedIbiFilePath,'afterChangesDB');
    end
    
    
    clear ecgAddedIbiData
    clear afterChangesDB
    %ecgAddedIbiData.changeDB.toRemTimes(end) = ecgtimecorrections1.toRemTimes(i);

%     for i=1:height(ecgTimeCorrSub)
%         if ~isnan(ecgtimecorrections1.toRemTimes(i))
%             pIdx = ecgtimecorrections1.currSubs(i);
%             ecgAddedIbiFileName = strcat('add_',...
%                 paths.fileNameMatchTable.Mkr_ECG_name(pIdx),...
%                 '.mat');
%             ecgAddedIbiFilePath = ...
%                 fullfile(paths.ecgAddedIbiDirPath,ecgAddedIbiFileName);
%             if exist(ecgAddedIbiFilePath,'file')
%                 ecgAddedIbiData = load(cell2mat(ecgAddedIbiFilePath));
%             end
%             toRemTimesCurrSub = ecgtimecorrections1.toRemTimes(i);
%             
%             %ecgAddedIbiData.changeDB.toRemTimes(end) = ecgtimecorrections1.toRemTimes(i);
%         end
end
%%

