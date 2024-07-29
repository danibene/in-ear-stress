function [varargout] = readecg(paths,pIdx,varargin)
% Read ECG data
%
% IN		paths                   struct containing file paths
%
%           pIdx                    numerical participant ID
%
%			times                	beginning and end time of segment to be 
%                                   read, in seconds (default: entire
%                                   recording)
%
%           beatType                option to specify types of heartbeats 
%                                   that are included (default: only sinus)
%
%           timeRef                 option to specify the signal used as
%                                   a reference for the timestamps
%                                   (default: start of ARP recording)
%
%           output                  option to specify output format
%
% OUT		varargout               default interbeat intervals and their 
%                                   respective timestamps, but can select
%                                   entire .mat file with interbeat
%                                   interval timing, raw ECG, or a
%                                   uniformly sampled time vector with
%                                   heartbeat timestamps represented as 1s
%                                   (for easy alignment with audio)
%
% v20211217 DB
    
    % parse inputs
    defaultBeatType = 'sinus';
    defaultTimes = [];
    defaultTimeRef = 'arp';
    defaultOutput = 'ibi';
    defaultChanNum = [1,2];
    defaultAssumedFs = [];
    expectedBeatTypes = {'allvalid','sinus','ignoreadd'};
    expectedTimeRef = {'arp','ecg'};
    expectedOutput = {'ibi','mat','raw','hbas1'};

    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    validList = @(x) (isa(x,'double') && length(x)==2) || isempty(x);
    addRequired(p,'paths');
    addRequired(p,'pIdx',validScalarPosNum);
    addOptional(p,'times',defaultTimes,validList);
    addParameter(p,'beatType',defaultBeatType,...
                 @(x) any(validatestring(x,expectedBeatTypes)));
    addParameter(p,'timeRef',defaultTimeRef,...
        @(x) any(validatestring(x,expectedTimeRef)));
    addParameter(p,'output',defaultOutput,...
        @(x) any(validatestring(x,expectedOutput)));
    addParameter(p,'chanNum',defaultChanNum)%,...
        %validScalarPosNum);
    addParameter(p,'assumedFs',defaultAssumedFs);
    
    parse(p,paths,pIdx,varargin{:});

    paths = p.Results.paths; 
    pIdx = p.Results.pIdx;
    times = p.Results.times;
    beatType = p.Results.beatType;
    timeRef = p.Results.timeRef;
    output = p.Results.output;
    chanNum = p.Results.chanNum;
    assumedFs = p.Results.assumedFs;
    
    ecgFileName = strcat(paths.fileNameMatchTable.Mkr_ECG_name(pIdx),...
        '.mat');
    ecgFilePath = fullfile(paths.ecgDirPath,ecgFileName);
    ecgData=load(cell2mat(ecgFilePath));
    
    ecgAddedIbiFileName = strcat('add_',...
        paths.fileNameMatchTable.Mkr_ECG_name(pIdx),...
        '.mat');
    ecgAddedIbiFilePath = ...
        fullfile(paths.ecgAddedIbiDirPath,ecgAddedIbiFileName);
    
    % sampling frequency = 500
    fs = ecgData.Param.AtoD.Fech;
    
    if strcmp(output,'raw')
        ecgRawFileName = ecgData.Param.Protocol.File_Raw;
        ecgRawFilePath = fullfile(paths.ecgRawDirPath,ecgRawFileName);
        f=fopen(ecgRawFilePath,'r');
        dat=fread(f,'uint16');
        fclose(f);
        totalNumChan = 3;
        allEcgSig = NaN(length(dat)/totalNumChan,length(chanNum));
        for chIdx=1:length(chanNum)
            allEcgSig(:,chIdx) = dat(chanNum(chIdx):totalNumChan:end);
        end
        allEcgSigTime = ((1:(length(dat)/totalNumChan))/fs).';
        allDataPoints = allEcgSig;
        allTimePoints = allEcgSigTime;
    else
        % by default only including sinus beats since frequency-domain HRV
        % features should be computed with only sinus beats, and there are so
        % few non-sinus beats just removed from all further analyses to
        % simplify
        if strcmp(beatType,'sinus')
            garde=find(ecgData.Mkr.vRejectU_RR==0 & ecgData.Mkr.vVSV==0);
        % take all the valid beats except for the one that was added manually
        % (by default decided to ignore since it seems to be a non-sinus beat)
        elseif strcmp(beatType,'ignoreadd')
            garde=find(ecgData.Mkr.vRejectU_RR==0 & ecgData.Mkr.vVSV~=20);
        % take all of the valid beats including one added manually
        else
            garde=find(ecgData.Mkr.vRejectU_RR==0 | ecgData.Mkr.vVSV==20);
        end

        % get the times of the peaks in seconds
        origIbiTime=ecgData.Mkr.t_Rcm(garde)/fs; 
        % corresponding interbeat intervals
        origIbi=ecgData.Mkr.vRR(garde); 

        % check if there are missing IBIs to be added
        if exist(ecgAddedIbiFilePath,'file')
            ecgAddedIbiData = load(cell2mat(ecgAddedIbiFilePath));
            ecgData.ecgAddedIbiData = ecgAddedIbiData;
            if isfield(ecgAddedIbiData,'afterChangesDB')
                allEcgIbiTime = ecgAddedIbiData.afterChangesDB.ibiTime;
                allEcgIbi = ecgAddedIbiData.afterChangesDB.ibi;
            else
                ibiTimeCombined = origIbiTime;
                ibiCombined = origIbi;
                if isfield(ecgAddedIbiData,'add')
                    for i = 1:length(ecgAddedIbiData.add)
                        ibiTimeCombined = [ibiTimeCombined,ecgAddedIbiData.add(i).tacq];
                        ibiCombined = [ibiCombined,ecgAddedIbiData.add(i).RR];
                    end   
                end
                [ibiTimeSorted,sortIdx] = sort(ibiTimeCombined);
                ibiSorted = ibiCombined(sortIdx);
                [~,ia,~] = unique(ibiTimeSorted,'stable');
                allEcgIbiTime = ibiTimeSorted(ia);
                allEcgIbi = ibiSorted(ia);
            end
        else
            % get the times of the peaks in seconds
            allEcgIbiTime = origIbiTime; 
            % corresponding interbeat intervals
            allEcgIbi = origIbi; 
        end
        diffIbiTime = diff(allEcgIbiTime);

        % check if the minimum difference between IBI times and the minimum IBI
        % matches
        % otherwise there might be duplicates;
        if min(diffIbiTime) > min(allEcgIbi)
            warning("The minimum difference between IBI times and " + ...
                "the minimum IBI do not match")
        end
        allDataPoints = ensurecolumn(allEcgIbi);
        allTimePoints = ensurecolumn(allEcgIbiTime);
    end
    
    if ~(isempty(assumedFs) || assumedFs == fs)
        oldFs = fs;
%         oldPossTimes = ((floor(min(allTimePoints)):...
%             ceil(max(allTimePoints))*oldFs)/oldFs).';
%         assumedPossTimes = ((floor(min(allTimePoints)):...
%             ceil(max(allTimePoints*(oldFs/assumedFs)))*assumedFs)/assumedFs).';
% 
%         timeIdx = NaN(size(allTimePoints));
%         for mkrIdx = 1:length(allTimePoints)
%             [~,timeIdx(mkrIdx)] = min(abs(oldPossTimes - ...
%                 allTimePoints(mkrIdx)));
%         end

        %%
        % allTimePoints = assumedPossTimes(timeIdx);
        allTimePoints = allTimePoints.*(oldFs/assumedFs);
        
        if ~strcmp(output,'raw')
            % recalculate IBI assuming new fs
            allDataPoints = allDataPoints.*(oldFs/assumedFs);
        end

    end
    
    % if the timestamps should be in reference to the start of the ARP
    % recording then the delay between ECG and ARP is added to the times in
    % reference to the start of the ECG recoridng
    if strcmp(timeRef,'arp')
        if ~(isempty(assumedFs) || assumedFs == fs)
            delayArpRefEcg = finddelayecg2arp(paths,pIdx,...
                'assumedFs',assumedFs,'saveFile','loadWithoutSave');
        else
            delayArpRefEcg = finddelayecg2arp(paths,pIdx);
        end
        timeToAddRefEcg = delayArpRefEcg;
    else
        timeToAddRefEcg = 0;
    end
    
    % if no times are specified return all the data for this recording
    if isempty(times)
        startTimeRefEcg = -Inf;
        endTimeRefEcg = Inf;
    else
        startTimeRefEcg = times(1) + timeToAddRefEcg;
        endTimeRefEcg = times(2) + timeToAddRefEcg;
    end
    
    segData = allDataPoints((allTimePoints > ...
        startTimeRefEcg & ...
        allTimePoints <= endTimeRefEcg),:);
    segTime = allTimePoints((allTimePoints > ...
        startTimeRefEcg & ...
        allTimePoints <= endTimeRefEcg)) - ...
        timeToAddRefEcg;
    
    if strcmp(output,'hbas1')
    succDiffIbiThresh = 1/1000;
    succDiffIdx = (abs(diff(segTime) - segData(2:end)./1000) ...
        < succDiffIbiThresh);
        if min(succDiffIdx)<1
            warning("There are missing IBIs; will interpolate to " + ...
            "estimate the timestamps of the missing heartbeats")
            desiredFs = 2;
            interplMethod = 'spline';
            [interplIbi,interplIbiTime] = ...
                interplibi(segData,segTime,desiredFs,interplMethod);
            [hbTime] = interplibi2peaktime(interplIbi,interplIbiTime,...
                segTime(1) - segData(1)/1000);
        else
            hbTime = [segTime(1) - segData(1)/1000; segTime];
        end
        desiredFsHbAsOnes = 8000;
        
        if times(1) >= 0
            segTime = ((1:ceil(hbTime(end) * ...
                desiredFsHbAsOnes))/desiredFsHbAsOnes).';
            hbTime = hbTime(hbTime >= 0);
        else
            segTime = ((1:ceil((hbTime(end) - hbTime(1)) * ...
                desiredFsHbAsOnes))/desiredFsHbAsOnes).' + hbTime(1);
        end
        
        segData = zeros(size(segTime));
        
        for hb=1:length(hbTime)
            [~,idx] = min(abs(hbTime(hb)-segTime));
            segData(idx) = 1;
        end
    end

    
    % by default returns the interbeat intervals and their respective
    % timestamps, but can also specify the entire .mat file
    if strcmp(output,'mat')
        varargout{1} = ecgData;
    else
        varargout{1} = segData;
        varargout{2} = segTime;
    end
end