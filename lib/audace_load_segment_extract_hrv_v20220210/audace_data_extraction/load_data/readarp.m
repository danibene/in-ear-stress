function [varargout] = readarp(paths,pIdx,varargin)
% Read a single channel of ARP data
%
% IN		paths                   struct containing file paths
%
%           pIdx                    numerical participant ID
%
%			times                	beginning and end time of segment to be 
%                                   read, in seconds (default: entire
%                                   recording)
%
%           timeRef                 option to specify the signal used as
%                                   a reference for the timestamps
%                                   (default: start of ARP recording)
%
%           output                  option to specify output
%
% OUT		varargout               default raw data
%
% v20211217 DB
    
    % parse inputs
    defaultSigName = {'IEML','IEMR','OEML','OEMR'};
    defaultTimes = [];
    defaultTimeRef = 'arp';
    defaultOutput = 'rawrec';
    expectedSigNames = {'IEML','IEMR','OEML','OEMR','PPG'};
    expectedTimeRef = {'arp','ecg'};
    expectedOutput = {'rawrec','rawfit'};

    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    validList = @(x) (isa(x,'double') && length(x)==2) || isempty(x);
    validSigNames = @(x) validateStringParameter(expectedSigNames,x);

    addRequired(p,'paths');
    addRequired(p,'pIdx',validScalarPosNum);
    addOptional(p,'times',defaultTimes,validList);
    addParameter(p,'sigName',defaultSigName,...
        validSigNames);
                % @(x) any(validatestring(x,expectedSigNames)));
    addParameter(p,'timeRef',defaultTimeRef,...
        @(x) any(validatestring(x,expectedTimeRef)));
    addParameter(p,'output',defaultOutput,...
        @(x) any(validatestring(x,expectedOutput)));
    
    parse(p,paths,pIdx,varargin{:});

    paths = p.Results.paths; 
    pIdx = p.Results.pIdx;
    times = p.Results.times;
    sigName = p.Results.sigName;
    timeRef = p.Results.timeRef;
    output = p.Results.output;
        
    chanDirPath = cell2mat(fullfile(paths.arpDirPath,paths.fileNameMatchTable.Audio_name(pIdx),...
        'Channels'));
    chanFileName = dir(fullfile(chanDirPath,'*.json')).name;
    chanFilePath = fullfile(chanDirPath,chanFileName);
    chIdx = jsondecode(fileread(chanFilePath));
    fields = fieldnames(chIdx);
    possFieldIdx = NaN(1,length(fields));
    for fIdx=1:length(fields)
        possFieldIdx(fIdx) = chIdx.(fields{fIdx});
    end
    if iscell(sigName)
        relChIdx = [];
        % to preserve order in cell
        for sigIdx=1:length(sigName)
            selIdx = possFieldIdx(contains(fields,sigName{sigIdx}));
            if ~isempty(selIdx)
                relChIdx(end+1:end+length(selIdx)) = selIdx;
            end
        end
    else
        relChIdx = possFieldIdx(contains(fields,sigName));
    end
    
    if strcmp(output,"rawfit")
        fitDirPath = cell2mat(fullfile(dataPaths.arpDirPath,...
            dataPaths.fileNameMatchTable.Audio_name(pIdx),...
                'FitCheck'));
        d = dir(fitDirPath);
        subdirList = fullfile({d.folder}', {d.name}'); 
        subdirList(~[d.isdir]) = []; %remove non-directories
        for i = 1:length(subdirList)
            fileList = dir(fullfile(subdirList{i}, '*.wav')); 
            if i==1
                fitFile = fileList;
            elseif ~isempty(fileList)
                fitFile(end+1:end+length(fileList)) = fileList;
            end
        end
        %%
        fitLabel = "";
        fitLabel(1) = [];
        %%
        for fIdx=1:length(fitFile)
            fitLabel(fIdx) = extractBetween(fitFile(fIdx).name,"fit-check-",".wav");
            fitNumStr = extractAfter(fitLabel(fIdx),"run-");
            if isempty(str2num(fitNumStr))
                fitNum(fIdx) = NaN;
            else
                fitNum(fIdx) = str2num(fitNumStr);
            end
        end
        %%
        if strcmp(fitname,"initial")
            [~,I] = max(fitNum,[],'omitnan');
            fIdx = I;
        else
            idcs = (1:length(fitNum));
            fIdx = idcs(isnan(fitNum));
        end
        fitFilePath = fullfile(fitFile(fIdx).folder,fitFile(fIdx).name);
        info = audioinfo(fitFilePath);
        allCh = audioread(fitFilePath);
        segData = allCh(:,relChIdx);
        fs = info.SampleRate;
        segTime = ((1:info.TotalSamples)/fs).';
    else
        arpDirPath = cell2mat(fullfile(paths.arpDirPath,paths.fileNameMatchTable.Audio_name(pIdx),...
            'Recordings'));
        arpFileName = dir(fullfile(arpDirPath,'*.wav')).name;
        arpFilePath = fullfile(arpDirPath,arpFileName);
        info = audioinfo(arpFilePath);
        fs = info.SampleRate;

        if strcmp(timeRef,'arp')
            timesRefArp = times;
            timeToAddRefArp = 0;
        else
            delayArpRefEcg = finddelayecg2arp(paths,pIdx);
            timeToAddRefArp = (-1)*delayArpRefEcg;
        end


        if isempty(timesRefArp)
            allCh = audioread(arpFilePath);
            segSigTime = ((1:info.TotalSamples)/fs).';
        else
            firstSamp = ceil(timesRefArp(1)*fs);
            if firstSamp < 1
                firstSamp = 1;
            end
            lastSamp = floor(timesRefArp(2)*fs);
            allCh = audioread(arpFilePath,[firstSamp,lastSamp]);
            segSigTime = ((firstSamp:lastSamp)/fs).';
        end

        segSig = allCh(:,relChIdx); 

        segData = segSig;
        segTime = segSigTime + timeToAddRefArp;
    end
    if contains(output,'raw')
        varargout{1} = segData;
        varargout{2} = segTime;
    end
    
    function validateStringParameter(expectedNames,x)
        if iscell(x)
            for xIdx=1:length(x)
                any(validatestring(x{xIdx},expectedNames));
            end
        else
            any(validatestring(x,expectedNames));
        end
    end

end