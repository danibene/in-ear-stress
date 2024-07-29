function [varargout] = timehrv(ibi,varargin)
    
    if isempty(ibi)
        ibi = NaN(3,1);
    end

    defaultIbiTime = cumsum(ibi)./1000;
    defaultFeatNames = {'MeanNN','SDNN','RMSSD','MinNN','MaxNN'};
    defaultTimeMissing= [];
    defaultIbiBaseline = NaN;
    defaultOutput = "featvalues";

    p = inputParser;
    addRequired(p,'ibi');
    addOptional(p,'ibiTime',defaultIbiTime);
    addParameter(p,'featNames',defaultFeatNames);
    addParameter(p,'timeMissing',defaultTimeMissing);
    addParameter(p,'ibiBaseline',defaultIbiBaseline);
    addParameter(p,'output',defaultOutput)
    
    parse(p,ibi,varargin{:});

    ibi = p.Results.ibi; 
    ibiTime = p.Results.ibiTime;
    featNames = p.Results.featNames;
    timeMissing = p.Results.timeMissing;
    ibiBaseline = p.Results.ibiBaseline;
    output = p.Results.output;
    
    meanNN = mean(ibi);
    SDNN = std(ibi);
    minNN = min(ibi);
    maxNN = max(ibi);
    medNN = median(ibi);

    x = ibi(1:end-1);
    y = ibi(2:end);

    yTime = ibiTime(2:end);
    yOnlySucc = y;
    yOnlySucc(ismember(yTime,timeMissing)) = NaN;
    
    succDiffWithNan = yOnlySucc - x;
    succDiff = succDiffWithNan(~isnan(succDiffWithNan));

    RMSSD = rms(succDiff);

    NN50 = length(succDiff(succDiff>50));
    PNN50 = (length(succDiff(succDiff>50))/length(ibi))*100;

    NN20 = length(succDiff(succDiff>20));
    PNN20 = (length(succDiff(succDiff>20))/length(ibi))*100;
    
    minNNminusBaselineNN = minNN - ibiBaseline;
    maxNNminusBaselineNN = maxNN - ibiBaseline;
    possibleFeatNames = {'MeanNN','SDNN','RMSSD','MinNN','MaxNN',...
        'NN50','PNN50','NN20','PNN20',...
        'MinNNminusBaselineNN','MaxNNminusBaselineNN','MedNN'};
    possibleValues = [meanNN,SDNN,RMSSD,minNN,maxNN,...
        NN50,PNN50,NN20,PNN20,minNNminusBaselineNN,maxNNminusBaselineNN,...
        medNN];

    [~,newOrder] = ismember(possibleFeatNames,featNames);
    possIdx = 1:length(possibleValues);
    outputIdx = possIdx(newOrder~=0);
    outVec = possibleValues(outputIdx);

    if strcmp(output,"featvalues")
        varargout{1} = outVec;
    else
        varargout{1} = possibleValues;
        varargout{2} = possibleFeatNames;
    end
end
