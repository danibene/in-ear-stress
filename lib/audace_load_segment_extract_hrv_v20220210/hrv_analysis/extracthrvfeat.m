function [varargout] = extracthrvfeat(ibi,varargin)

    defaultIbiTime = cumsum(ibi)./1000;
    defaultIbiBaseline = NaN;
    defaultSmoothFc = 0.15;
    defaultFeatNames = {'MeanNN','SDNN','RMSSD','MinNN','MaxNN',...
        'MinNN_smoothed0p15','MaxNN_smoothed0p15','RatioLFHF',...
        'LF', 'HF', 'PeakLF', 'PeakHF'};
    % TODO: add warning if the selected smoothFc 
    % does not match the feature names
    defaultTimeRange = []; 
    defaultNormSub = NaN;
    defaultNormDiv = NaN;
    defaultNormMult = 1;%0.6745;
    defaultNorm = 0;
    defaultIncludeLn = 0;

    p = inputParser;
    addRequired(p,'ibi');
    addOptional(p,'ibiTime',defaultIbiTime);
    addParameter(p,'ibiBaseline',defaultIbiBaseline);
    addParameter(p,'smoothFc',defaultSmoothFc);
    addParameter(p,'featNames',defaultFeatNames);
    addParameter(p,'timeRange',defaultTimeRange);
    addParameter(p,'normDiv',defaultNormSub);
    addParameter(p,'normSub',defaultNormDiv);
    addParameter(p,'normMult',defaultNormMult);
    addParameter(p,'norm',defaultNorm);
    addParameter(p,'includeLn',defaultIncludeLn);
    
    parse(p,ibi,varargin{:});

    ibi = p.Results.ibi; 
    ibiTime = p.Results.ibiTime;
    ibiBaseline = p.Results.ibiBaseline;
    smoothFc = p.Results.smoothFc;
    featNames = p.Results.featNames;
    timeRange = p.Results.timeRange;
    normSub = p.Results.normSub;
    normDiv = p.Results.normDiv;
    normMult = p.Results.normMult;
    norm = p.Results.norm;
    includeLn = p.Results.includeLn;

    enoughData = 1;
    
    % at least 6 data points required for some features (filtering)
    if length(ibi) < 6
        enoughData = 0;
    else
        y = ibi(2:end);
        shouldIbi = diff(ibiTime)*1000;
        acceptDiffTrueShouldIbi = 0.000001;
        shouldIbiTime = ibiTime(2:end);
        discontinuityIdx = (abs(y-shouldIbi) > ...
            acceptDiffTrueShouldIbi);
        discontinuityTime = shouldIbiTime(discontinuityIdx);
        totalSecsIbi = sum(ibi)/1000;
        if isempty(timeRange)
            segSecs = max(ibiTime) - min(ibiTime);
        else
            segSecs = diff(timeRange);
        end
        acceptPercTotalSecsMissing = 50;
        if totalSecsIbi < segSecs*(acceptPercTotalSecsMissing/100)
            enoughData = 0;
        end
    end
    
    if norm
        ibi = (normMult*(ibi - normSub))./normDiv;
        ibiBaseline = 0;
    end
    
    [~,timeFeatNames] = timehrv([],'output','featnames');
%         {'MeanNN','SDNN','RMSSD','MinNN','MaxNN',...
%         'NN50','PNN50','NN20','PNN20',...
%         'MinNNminusBaselineNN','MaxNNminusBaselineNN','MedNN',...
%         'AreaAbove_RefBaseline','AreaBelow_RefBaseline','AreaI_RefBaseline',...
%         'Area_RefBaseline','AreaAbove_RefMean','AreaBelow_RefMean',...
%         'AreaI_RefMean','Area_RefMean','AreaAbove_RefMed','AreaBelow_RefMed',...
%         'AreaI_RefMed','Area_RefMed'};
    
    freqFeatNames = {'ULF','VLF','LF','HF','VHF','UHF'...
                'RatioLFHF','PeakLF','PeakHF'};
       
    nonlinearFeatNames = {'SD1', 'SD2', 'SD1SD2', 'SD2d','SD2a',...
        'SD2dRefBaseline','SD2aRefBaseline',...
        'SD2IRefBaseline',...
        'SD2I','C2d','C2a',...
        'SDNNd','SDNNa','SDNN_nonlinear','Cd','Ca',...
        'GI','SI','AI','PI','C1d','C1a',...
        'CorDim','ApEn'};

    [~,devBaseFeatNames] = devbasehrv([],'output','featnames');

    
    allFeatNames = [timeFeatNames, freqFeatNames, nonlinearFeatNames, devBaseFeatNames];
    
    if strcmp(featNames,"all")
        useAllFeats = 1;
        featNames = allFeatNames;
    else
        useAllFeats = 0;
    end
    
    selTimeFeatNames = timeFeatNames(ismember(timeFeatNames,featNames));
    selFreqFeatNames = freqFeatNames(ismember(freqFeatNames,featNames));
    selNonlinearFeatNames = ...
        nonlinearFeatNames(ismember(nonlinearFeatNames,featNames));
    selDevBaseFeatNames = ...
        devBaseFeatNames(ismember(devBaseFeatNames,featNames));
    selFeatNamesPreSmooth = [selTimeFeatNames,selFreqFeatNames,...
        selNonlinearFeatNames,selDevBaseFeatNames];
    if enoughData
        if ~isempty(selTimeFeatNames)
            timeOutVec = ...
                timehrv(ibi,ibiTime,'featnames',selTimeFeatNames,...
                'timeMissing',discontinuityTime,'ibiBaseline',ibiBaseline);
        else
            timeOutVec = [];
        end
        
        if ~isempty(selFreqFeatNames)
            freqOutVec = freqhrv(selFreqFeatNames,ibi,ibiTime);
        else
            freqOutVec = [];
        end
        
        if ~isempty(selNonlinearFeatNames)
            nonlinearOutVec = ...
                nonlinearhrv(nonlinearFeatNames,ibi,ibiTime,...
                discontinuityTime,ibiBaseline);
        else
            nonlinearOutVec = [];
        end

        if ~isempty(selDevBaseFeatNames)
            devBaseOutVec = ...
                devbasehrv(ibi,ibiTime,'featnames',selDevBaseFeatNames,...
                'timeMissing',discontinuityTime,'ibiBaseline',ibiBaseline);
        else
            devBaseOutVec = [];
        end

        allOutVec = [timeOutVec, freqOutVec, nonlinearOutVec, devBaseOutVec];
    else
        allOutVec = NaN(size([selTimeFeatNames,selFreqFeatNames,selNonlinearFeatNames,selDevBaseFeatNames]));
    end
    allOutVecNames = selFeatNamesPreSmooth;
    for fcIdx=1:length(smoothFc)
        
        L = char(num2str(smoothFc(fcIdx)));
        L(L=='.')='p';
        smoothedTimeFeatNames = strcat(timeFeatNames,'_smoothed',L);  
        
        smoothedFreqFeatNames = strcat(freqFeatNames,'_smoothed',L);
        
        smoothedNonlinearFeatNames = strcat(nonlinearFeatNames,'_smoothed',L);

        smoothedDevBaseFeatNames = strcat(devBaseFeatNames,'_smoothed',L);
        
        allFeatNames = [allFeatNames, smoothedTimeFeatNames, ...
            smoothedFreqFeatNames, smoothedNonlinearFeatNames, ...
            smoothedDevBaseFeatNames];

        if useAllFeats
            selSmoothedTimeFeatNames = smoothedTimeFeatNames;
            selSmoothedFreqFeatNames = smoothedFreqFeatNames;
            selSmoothedNonlinearFeatNames = smoothedNonlinearFeatNames;
            selSmoothedDevBaseFeatNames = smoothedDevBaseFeatNames;
        else
            selSmoothedTimeFeatNames = ...
                smoothedTimeFeatNames(ismember(smoothedTimeFeatNames,featNames));
            selSmoothedFreqFeatNames = ...
                smoothedFreqFeatNames(ismember(smoothedFreqFeatNames,featNames));
            selSmoothedNonlinearFeatNames = ...
                smoothedNonlinearFeatNames(ismember(smoothedNonlinearFeatNames,...
                featNames));
            selSmoothedDevBaseFeatNames = ...
                smoothedDevBaseFeatNames(ismember(smoothedDevBaseFeatNames,...
                featNames));
        end

        
        if enoughData
            [smoothedIbi,smoothedIbiTime] = smoothibi(ibi,ibiTime,smoothFc(fcIdx));
            
            if ~isempty(selSmoothedTimeFeatNames)
                smoothedTimeOutVec = ...
                    timehrv(smoothedIbi,smoothedIbiTime,'featnames',timeFeatNames,...
                'timeMissing',discontinuityTime,'ibiBaseline',ibiBaseline);
                allOutVecNames = [allOutVecNames, smoothedTimeFeatNames];
            else
                smoothedTimeOutVec = [];
            end
            
            if ~isempty(selSmoothedFreqFeatNames)
                smoothedFreqOutVec = freqhrv(freqFeatNames,...
                    smoothedIbi,smoothedIbiTime);
                allOutVecNames = [allOutVecNames, smoothedFreqFeatNames];
            else
                smoothedFreqOutVec = [];
            end
            
            if ~isempty(selSmoothedNonlinearFeatNames)
                smoothedNonlinearOutVec = nonlinearhrv(nonlinearFeatNames,...
                    smoothedIbi,smoothedIbiTime,discontinuityTime,ibiBaseline);
                allOutVecNames = [allOutVecNames, smoothedNonlinearFeatNames];
            else
                smoothedNonlinearOutVec = [];
            end

            if ~isempty(selSmoothedDevBaseFeatNames)
                smoothedDevBaseOutVec = ...
                    devbasehrv(smoothedIbi,smoothedIbiTime,'featnames',devBaseFeatNames,...
                'timeMissing',discontinuityTime,'ibiBaseline',ibiBaseline);
                allOutVecNames = [allOutVecNames, smoothedDevBaseFeatNames];
            else
                smoothedDevBaseOutVec = [];
            end
            allOutVec = [allOutVec, smoothedTimeOutVec, ...
                smoothedFreqOutVec, smoothedNonlinearOutVec, ...
                smoothedDevBaseOutVec];
        else
            allOutVec = NaN(size(allOutVecNames));
        end
        
    end

    if includeLn
        allOutVec = [allOutVec, mylog(allOutVec), myyeojohnson(allOutVec)];
        allFeatNames = [allFeatNames,strcat("Ln",allFeatNames),strcat("Yj",allFeatNames)];
        allOutVecNames = [allOutVecNames,strcat("Ln",allOutVecNames),strcat("Yj",allOutVecNames)];
    end

    if useAllFeats
        featNames = allFeatNames;
    end
    for i = 1:length(allOutVecNames)
       out.(allOutVecNames{i}) = allOutVec(i);
       
    end
    
    outVec = NaN(size(featNames));
    for i = 1:length(featNames)
       outVec(i) = out.(featNames{i}); 
    end
    varargout{1} = outVec;
    varargout{2} = featNames;
end