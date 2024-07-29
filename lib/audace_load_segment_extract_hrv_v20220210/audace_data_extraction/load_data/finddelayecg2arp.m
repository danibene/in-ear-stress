function [varargout] = finddelayecg2arp(paths,pIdx,varargin)
% Find delay between ECG and ARP data with Holter markers and tones
%
% IN		paths                   struct containing file paths
%           pIdx                    numerical participant ID
%			saveFile                option to save new file with delay time
%           output                  option to specify output
%
% OUT		varargout               default only scalar delay value, but
%                                   can also include table with individual 
%                                   marker times
%
% v20211213 DB
    
    % parse inputs
    defaultSaveFile = 'saveNewFile';
    defaultOutput = 'delay';
    defaultFs = 500;
    defaultAssumedFs = [];
    expectedSaveFile = {'readFromFile','saveNewFile','loadWithoutSave'};
    expectedOutput = {'delay','indMkrTable'};

    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    addRequired(p,'paths');
    addRequired(p,'pIdx',validScalarPosNum);
    addParameter(p,'saveFile',defaultSaveFile,...
                 @(x) any(validatestring(x,expectedSaveFile)));
    addParameter(p,'output',defaultOutput,...
        @(x) any(validatestring(x,expectedOutput)));
    addParameter(p,'fs',defaultFs)
    addParameter(p,'assumedFs',defaultAssumedFs)
    
    parse(p,paths,pIdx,varargin{:});
    paths = p.Results.paths; 
    pIdx = p.Results.pIdx;
    saveFile = p.Results.saveFile;
    output = p.Results.output;
    fs = p.Results.fs;
    assumedFs = p.Results.assumedFs;
    
    filePath = fullfile(paths.syncTimestampsDirPath,...
        strcat(paths.fileNameMatchTable.Audio_name(pIdx),...
        paths.syncTimestampsFileSuffix));
    if strcmp(saveFile,"readFromFile") && ...
            exist(filePath,'file')
        delayArpRefEcg = jsondecode(fileread(filePath));
    else
        %% initial alignment
        % how many seconds the ARP markers should be delayed by (for no delay = 0)
        initialArpDelayRefEcg = 0;
        syncLabels = {'BASELINE1_MKR','MENTAL_MKR','BASELINE2_MKR','NOISE_MKR',...
            'BASELINE3_MKR','CPT_MKR'};

        %% read files in mkr directories
        % read from new ARP mkr files (with comments removed)
        readFromNewArpMkr = 1;
        %% iterate over each participant
        syncIdx = 1;
        arpMkrIdx = [];
        ecgMkrIdx = [];
        segMkrMatchTable = paths.mkrMatchTable(string(paths.mkrMatchTable.Participant)==char(paths.fileNameMatchTable.Audio_name{pIdx}),:);

        ecgMkrFilename = strcat(paths.fileNameMatchTable.Mkr_ECG_name{pIdx},'.mat');

        ecgData = load(fullfile(paths.ecgMkrDirPath,ecgMkrFilename));
        % convert from minutes to seconds
        ecgMkr = ecgData.Param.EVENTS.List*60;

        arpMkrFilename = strcat(paths.fileNameMatchTable.Audio_name{pIdx}, ...
            '_holter_markers.txt');
        if ~exist(fullfile(paths.arpMkr4syncDirPath,arpMkrFilename),'file')
            arpMkrFilename = strcat(paths.fileNameMatchTable.Audio_name{pIdx}, ...
            '_holter-markers.txt');
        end

        if ~exist(fullfile(paths.arpMkr4syncDirPath,arpMkrFilename),'file') && pIdx == 17
            arpMkrFilename = '17_H-ML_holter_markers.txt';
        end

        newArpMkrFilepath = strcat(paths.arpMkr4syncDirPath,'_without_notes');
        if isfolder(newArpMkrFilepath) == 0
            mkdir(newArpMkrFilepath)
        end 

        newArpMkrFilename = strcat(arpMkrFilename(1:end-4),'_without_notes.txt');

        % There were sometimes notes in the txt files followed by blank lines
        % so this is a workaround to read the file as a table anyway
        fid = fopen(fullfile(paths.arpMkr4syncDirPath,arpMkrFilename),'r');
        sepByLine = textscan(fid,'%s','Delimiter','\n');
        fout = fopen(fullfile(newArpMkrFilepath,newArpMkrFilename), 'wt');
        firstEmptyLineFoundOrDone = 0;
        i = 0;
        firstEmptyIdx = 0;
        while ~firstEmptyLineFoundOrDone
            i = i + 1;
            if i == length(sepByLine{1})
                firstEmptyLineFoundOrDone = 1;
            end
            if ~contains(sepByLine{1,1}{i,1}, sprintf('\t'))
                firstEmptyIdx = i;
                firstEmptyLineFoundOrDone = 1;
            else
                fprintf(fout, '%s\n', sepByLine{1}{i});
            end
        end
        fclose(fid);
        fclose(fout);
        if firstEmptyIdx == 0
            arpMkr = readtable(fullfile(paths.arpMkr4syncDirPath,arpMkrFilename));
        elseif readFromNewArpMkr == 1
            arpMkr = readtable(fullfile(newArpMkrFilepath,newArpMkrFilename));
        else
            fid = fopen(fullfile(paths.arpMkrDirPath,arpMkrFilename),'r');
            sepByTab = textscan(fid,'%s','delimiter','\t');
            fclose(fid);
            nCellsPerLine = 3;
            arpMkrFlatWithNotes = sepByTab{1};
            arpMkrFlat = arpMkrFlatWithNotes(1:nCellsPerLine*(firstEmptyIdx - 1));
            arpMkrStr = cell2table(reshape(arpMkrFlat, ...
                [nCellsPerLine,firstEmptyIdx - 1]).');
            for cellIdx=1:length(arpMkrStr.Var1)
                arpMkrVar1(cellIdx) = str2double(arpMkrStr.Var1{cellIdx});
                arpMkrVar2(cellIdx) = str2double(arpMkrStr.Var2{cellIdx});
            end
            arpMkr.Var1 = arpMkrVar1.';
            arpMkr.Var2 = arpMkrVar2.';
            arpMkr.Var3 = arpMkrStr.Var3;
        end
        while any([arpMkrIdx==0, ecgMkrIdx==0, isempty(arpMkrIdx), isempty(ecgMkrIdx)])
            %arpMkrIdx==0 | ecgMkrIdx==0 | isempty(arpMkrIdx) | isempty(ecgMkrIdx)
            arpMkrLabel = syncLabels(syncIdx);
            ecgMkrLabel = syncLabels(syncIdx);
            arpMkrIdx = segMkrMatchTable.ArpMkr(strcmp(string(segMkrMatchTable.Label),string(arpMkrLabel)));
            ecgMkrIdx = segMkrMatchTable.EcgMkr(strcmp(string(segMkrMatchTable.Label),string(ecgMkrLabel)));
            if any([arpMkrIdx==0, ecgMkrIdx==0, isempty(arpMkrIdx), isempty(ecgMkrIdx)])
                syncIdx=syncIdx + 1;
            end                
        end

        onsetArpMkrRefEcgStart = arpMkr.Var1 + ecgMkr(ecgMkrIdx) - ...
            (arpMkr.Var1(arpMkrIdx) - arpMkr.Var1(1)) + ...
            initialArpDelayRefEcg - arpMkr.Var1(1);
        %% create new table with empty fields
        indMkrTable = segMkrMatchTable;

        indMkrTable.TimeOnsetArpMkrRefArpStart = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.TimeOffsetArpMkrRefArpStart = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.ArpMkrDuration = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.TimeOnsetArpMkrRefEcgStart = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.TimeEcgMkrRefEcgStart = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.DelayMkrsRefEcgStart = NaN(length(segMkrMatchTable.ArpMkr),1);

        indMkrTable.UserInputArpDelayRefEcg(1:length(segMkrMatchTable.ArpMkr)) = initialArpDelayRefEcg;
        indMkrTable.UserInputArpMkrIdx(1:length(segMkrMatchTable.ArpMkr)) = arpMkrIdx;
        indMkrTable.UserInputEcgMkrIdx(1:length(segMkrMatchTable.ArpMkr)) = ecgMkrIdx;

        indMkrTable.OptimalArpDelayRefEcg = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.OptimalTimeOnsetArpMkrRefEcgStart = NaN(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.OptimalDelayMkrsRefEcgStart = NaN(length(segMkrMatchTable.ArpMkr),1);

        indMkrTable.EcgMkrFilename = cell(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.EcgMkrFilename(:) = {ecgMkrFilename};
        indMkrTable.ArpMkrFilename = cell(length(segMkrMatchTable.ArpMkr),1);
        indMkrTable.ArpMkrFilename(:) = {arpMkrFilename};

        %% iterate over all of the markers per participant
        for i=1:length(segMkrMatchTable.ArpMkr)
            % if an ARP marker exists
            if segMkrMatchTable.ArpMkr(i)~=0 
                % the onset of the ARP marker relative to the start of the
                % audio recording
                indMkrTable.TimeOnsetArpMkrRefArpStart(i) = ...
                    arpMkr.Var1(segMkrMatchTable.ArpMkr(i));
                % the offset of the ARP marker relative to the start of the
                % audio recording
                indMkrTable.TimeOffsetArpMkrRefArpStart(i) = ...
                    arpMkr.Var2(segMkrMatchTable.ArpMkr(i));
                % the duration of the ARP marker (likely only one second if it
                % is supposed to correspond to the ECG marker but 
                % will be longer for tasks)
                indMkrTable.ArpMkrDuration(i) = ...
                    arpMkr.Var2(segMkrMatchTable.ArpMkr(i)) - ...
                    arpMkr.Var1(segMkrMatchTable.ArpMkr(i));
                % the onset of the ARP marker relative to the start of the ECG
                % recording
                indMkrTable.TimeOnsetArpMkrRefEcgStart(i) = ...
                    onsetArpMkrRefEcgStart(segMkrMatchTable.ArpMkr(i));
            end
            % if an ECG marker exists
            if segMkrMatchTable.EcgMkr(i)~=0
                % the ECG marker time relative to the start of the ECG
                % recording
                indMkrTable.TimeEcgMkrRefEcgStart(i) = ...
                    ecgMkr(segMkrMatchTable.EcgMkr(i));
                if ~(isempty(assumedFs) || assumedFs == fs)
                    indMkrTable.TimeEcgMkrRefEcgStart(i) = ...
                        indMkrTable.TimeEcgMkrRefEcgStart(i)*(fs/assumedFs);
                end
            end
            % if there is both an ARP and ECG marker corresponding to each
            % other
            if segMkrMatchTable.ArpMkr(i)~=0 && segMkrMatchTable.EcgMkr(i)~=0
                % calculate the delay in seconds between the markers 
                % given how they are synchronized now
                indMkrTable.DelayMkrsRefEcgStart(i) = ...
                    indMkrTable.TimeOnsetArpMkrRefEcgStart(i) - ...
                    indMkrTable.TimeEcgMkrRefEcgStart(i);  
            end
        end
        %% calculate the optimal synchronization between ARP and ECG recordings
        % in order to minimize the delay between individual markers
        OptimalArpDelayRefEcg = ...
            (-1)*mean(indMkrTable.DelayMkrsRefEcgStart,'omitnan') + ...
            initialArpDelayRefEcg;
        % list in new table
        indMkrTable.OptimalArpDelayRefEcg(:) = OptimalArpDelayRefEcg;
        %% calculate the new optimal ARP marker onset time
        OptimalTimeOnsetArpMkrRefEcgStart = arpMkr.Var1 + ecgMkr(ecgMkrIdx) ...
            - (arpMkr.Var1(arpMkrIdx) - arpMkr.Var1(1)) + ...
            OptimalArpDelayRefEcg - arpMkr.Var1(1);
        %%
        for i=1:length(segMkrMatchTable.ArpMkr)
            if segMkrMatchTable.ArpMkr(i)~=0
                % include new optimal ARP marker onset time in table
                indMkrTable.OptimalTimeOnsetArpMkrRefEcgStart(i) = ...
                    OptimalTimeOnsetArpMkrRefEcgStart(segMkrMatchTable.ArpMkr(i));
            end
            if segMkrMatchTable.ArpMkr(i)~=0 && segMkrMatchTable.EcgMkr(i)~=0
                % calculate the delay in seconds between the individual markers 
                % after optimizing the delay
                indMkrTable.OptimalDelayMkrsRefEcgStart(i) = ...
                    indMkrTable.OptimalTimeOnsetArpMkrRefEcgStart(i) - ...
                    indMkrTable.TimeEcgMkrRefEcgStart(i);
            end
        end
        %%
    delayWithNan = unique(indMkrTable.OptimalTimeOnsetArpMkrRefEcgStart - indMkrTable.TimeOnsetArpMkrRefArpStart);
    delayArpRefEcg = mean(delayWithNan,'omitnan');
    end
    if ~strcmp(saveFile,"loadWithoutSave")
        if ~isfolder(paths.syncTimestampsDirPath)
            mkdir(paths.syncTimestampsDirPath)
        end
        fileID = fopen(filePath,'w');
        fprintf(fileID,jsonencode(delayArpRefEcg));
        fclose(fileID);
    end
    
    if strcmp(output,'delay')
        varargout{1} = delayArpRefEcg;
    else
        varargout{1} = delayArpRefEcg;
        varargout{2} = indMkrTable;
    end
end