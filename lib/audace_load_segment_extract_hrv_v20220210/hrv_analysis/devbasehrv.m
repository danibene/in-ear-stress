function [varargout] = devbasehrv(ibi,varargin)
    
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

    
%     devBaselineFeatNames = ["devRefBaseline",...
%         "devDecRefBaseline",...
%         "devAccRefBaseline",...
%         "devPosRefBaseline",...
%         "devNegRefBaseline",...
%         "devRefMean",...
%         "devDecRefBaseline"];
    devBaseNames = ["",...
        "Dec",...
        "Acc",...
        "Pos",...
        "Neg"];
    devBaselineFeatNames = [strcat(strcat("dev",devBaseNames),"RefBaseline"),...
        strcat(strcat("dev",devBaseNames),"RefMean")];   

    %refNames = ["RefBaseline","RefMean","RefMed"];
    aucFeatNames = strcat(strcat("AUC",["posAcc","posDec","posCon","negAcc","negDec","negCon","pos","neg",""]),"RefBaseline");
    possibleFeatNames = [cellstr(devBaselineFeatNames),...
        cellstr(aucFeatNames)];

    if ~strcmp(output,'featnames')
        x = ibi(1:end-1);
        y = ibi(2:end);
        
        yTime = ibiTime(2:end);
        yOnlySucc = y;
        yOnlySucc(ismember(yTime,timeMissing)) = NaN;
        
        succDiffWithNan = yOnlySucc - x;
        succDiff = succDiffWithNan(~isnan(succDiffWithNan));

        succDiffX = succDiffWithNan(1:end-1);
        succDiffY = succDiffWithNan(2:end);
        
        succDiffYTime = yTime(2:end);
        succDiffYOnlySucc = succDiffY;
        succDiffYOnlySucc(ismember(succDiffYTime,timeMissing)) = NaN;

        succ2DiffWithNan = succDiffYOnlySucc - succDiffX;
        succ2Diff = succ2DiffWithNan(~isnan(succ2DiffWithNan));
        
        allIdx = (ibi > 0);
        decIdx = [false;(succDiffWithNan > 0)];
        accIdx = [false;(succDiffWithNan < 0)];
        conIdx = [false;(succDiffWithNan==0)];
        %%
        refs = [ibiBaseline,mean(ibi)];
        devBaselineFeatValues = NaN(size(devBaselineFeatNames));
        for rIdx=1:length(refs)
            ref = refs(rIdx);
            dev = sqrt(sum((ibi-ref).^2)/(length(ibi)-1));
            %devDecRefBaseline = sqrt(sum((ibi(decIdx)-ibiBaseline).^2)/(length(ibi(decIdx))-1));
            devDec = sqrt(sum((ibi(decIdx)-ref).^2)/(length(ibi)-1));
            %devAccRefBaseline = sqrt(sum((ibi(accIdx)-ibiBaseline).^2)/(length(ibi(accIdx))-1));
            devAcc = sqrt(sum((ibi(accIdx)-ref).^2)/(length(ibi)-1));
            devPos = sqrt(sum((ibi(ibi - ref > 0) - ...
                ref).^2)/(length(ibi)-1));
            devNeg = sqrt(sum((ibi(ibi - ref < 0) - ...
                ref).^2)/(length(ibi)-1));
            firstIdx = (rIdx-1)*length(devBaseNames) + 1;
            lastIdx = firstIdx + length(devBaseNames) - 1;
            devBaselineFeatValues(firstIdx:lastIdx) = ...
                [dev,devDec,devAcc,...
                devPos,devNeg];
        end
        
        %%
        desiredFs = 4;
        interplMethod = 'linear';
        [interplIbi,interplIbiTime] = interplibi(ibi,ibiTime,desiredFs,interplMethod);
        decIdxArea = logical(interp1(ibiTime,double(decIdx),interplIbiTime,'nearest'));
        accIdxArea = logical(interp1(ibiTime,double(accIdx),interplIbiTime,'nearest'));
        conIdxArea = logical(interp1(ibiTime,double(conIdx),interplIbiTime,'nearest'));
        
        %%
        N = length(interplIbiTime);
        a = interplIbiTime(1);
        b = interplIbiTime(end) + 1/desiredFs;
        % Number of divisions
        % If we divide the interval [a, b] into N pieces ,
        % each piece would have width
        width = ( b - a ) / N;
        %%
        if strcmp(output,'plot')
            figure
            plot(ibiTime,ibi)
            hold on
        end
        % width = 1/desiredFs;
        % The command below says to look at all values of x,
        % starting from a, increasing by the variable ’width ’
        % each time , until the final value where x = b - width
        posIdxArea = (interplIbi-ibiBaseline > 0);
        negIdxArea = (interplIbi-ibiBaseline < 0);
        idxArea = (abs(interplIbi - ibiBaseline) > 0);
        
        posAccIdxArea = (posIdxArea & accIdxArea);
        posDecIdxArea = (posIdxArea & decIdxArea);
        posConIdxArea = (posIdxArea & conIdxArea);
        negAccIdxArea = (negIdxArea & accIdxArea);
        negDecIdxArea = (negIdxArea & decIdxArea);
        negConIdxArea = (negIdxArea & conIdxArea);


        allIdxArea = {posAccIdxArea,posDecIdxArea,posConIdxArea,...
            negAccIdxArea,negDecIdxArea,negConIdxArea,posIdxArea,negIdxArea,...
            idxArea};
        
        colorMat = [[0.4940 0.1840 0.5560];...
            [0.4660 0.6740 0.1880];...
            [0.3010 0.7450 0.9330];...
            [0.6350 0.0780 0.1840];...
            [0.3847 0.2839 0.4834];
            [0.4398 0.9302 0.9429]];
        
        approxInteg = NaN(1,length(allIdxArea));
        for c=1:length(allIdxArea)
        % For each of the x values , we draw a rectangle
        % with the lower left corner at coordinate (x, 0) ,
        % width the variable ’width ’ , and height the value f(x).
            relIbi = interplIbi(allIdxArea{c});
            relIbiTime = interplIbiTime(allIdxArea{c});
            if strcmp(output,'plot')
                for i=1:length(relIbiTime)
                    if relIbi(i)-ibiBaseline > 0
                        rectangle ('Position', [relIbiTime(i) ibiBaseline ...
                            width abs(relIbi(i)-ibiBaseline)] , 'EdgeColor' , colorMat(c,:));
                        hold on
                    else
                        rectangle ('Position', [relIbiTime(i) relIbi(i) width ...
                            abs(relIbi(i)-ibiBaseline)] , 'EdgeColor' , colorMat(c,:));
                        hold on
                    end
                end
                l(c) = line(NaN,NaN,'LineWidth',2,'LineStyle','-','Color',colorMat(c,:));
            end
            approxInteg(c) = sum(abs(relIbi-ibiBaseline).* width);
        end
        if strcmp(output,'plot')
            legend(l,aucFeatNames)
            
            hold off % Done plotting
        end
        
        
        kurt = kurtosis(ibi,0);
        skew = skewness(ibi,0);
        
        sumStatValues = [kurt,skew];
        possibleValues = [devBaselineFeatValues,approxInteg];
        
    else
        possibleValues = NaN(size(possibleFeatNames));
    end
    
%     meanNN = mean(ibi);
%     SDNN = std(ibi);
%     minNN = min(ibi);
%     maxNN = max(ibi);
%     medNN = median(ibi);
% 
%     x = ibi(1:end-1);
%     y = ibi(2:end);
% 
%     yTime = ibiTime(2:end);
%     yOnlySucc = y;
%     yOnlySucc(ismember(yTime,timeMissing)) = NaN;
%     
%     succDiffWithNan = yOnlySucc - x;
%     succDiff = succDiffWithNan(~isnan(succDiffWithNan));
% 
%     devRefBaseline = sqrt(sum((ibi-ibiBaseline).^2)/(length(ibi)-1));
% 
%     RMSSD = rms(succDiff);
% 
%     NN50 = length(succDiff(succDiff>50));
%     PNN50 = (length(succDiff(succDiff>50))/length(ibi))*100;
% 
%     NN20 = length(succDiff(succDiff>20));
%     PNN20 = (length(succDiff(succDiff>20))/length(ibi))*100;
%     
%     minNNminusBaselineNN = minNN - ibiBaseline;
%     maxNNminusBaselineNN = maxNN - ibiBaseline;
% 
%     allIdx = (ibi > 0);
%     decIdx = [false;(succDiffWithNan > 0)];
%     accIdx = [false;(succDiffWithNan < 0)];
%         
%     AUCRefNames = ["Baseline","Mean","Med"];
%     AUCRefValues = [ibiBaseline,meanNN,medNN];
%     AUCIdxNames = [""];%,"a","d"];
%     AUCIdxValues = {allIdx};%,accIdx,decIdx};
% 
%     AUCBaseFeatNames = ["pAUC","nAUC","AUC","AUCI"];
%     AUCFeatNames = strings(1,length(AUCRefNames)*...
%         length(AUCIdxNames)*length(AUCBaseFeatNames));
%     AUCFeatValues = NaN(1,length(AUCRefNames)*...
%         length(AUCIdxNames)*length(AUCBaseFeatNames));
%     
%     for i=1:length(AUCRefNames)
% 
%         for j=1:length(AUCIdxValues)
% 
%             AUCFeatNames((1 + length(AUCBaseFeatNames)*(j-1) + ...
%                 length(AUCIdxValues)*length(AUCBaseFeatNames)*(i-1)) : ...
%                 (length(AUCBaseFeatNames)*(j) + ...
%                 length(AUCIdxValues)*length(AUCBaseFeatNames)*(i-1))) = ...
%                 strcat(AUCBaseFeatNames,AUCIdxNames(j),...
%                 "Ref",AUCRefNames(i));
% 
%             lin = max(ibi,AUCRefValues(i))-AUCRefValues(i);
%             lin(isnan(linWithNan)) = 0;
% 
%             if length(lin) > 1
%                 AUC_Each_Pt_CumSum = ...
%                     -cumtrapz(ibiTime, lin);
%                 AUC_Each_Pt = [AUC_Each_Pt_CumSum(1);...
%                     diff(AUC_Each_Pt_CumSum)];
%                 relAUC = cumsum(AUC_Each_Pt(AUCIdxValues{j}));
%                 AUCFeatValues(1 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     trapz(relIbiTime, lin);
%             elseif length(lin) == 1
%                 AUCFeatValues(1 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     0;
%             end
%             
%             linWithNan = min(relIbi,AUCRefValues(i))-AUCRefValues(i);
%             lin = linWithNan(~isnan(linWithNan));
%             relIbiTime = relIbiTimeWithNaN(~isnan(linWithNan));
% 
%             if length(lin) > 1
%                 AUCFeatValues(2 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     -trapz(relIbiTime,lin);   
%             elseif length(lin) == 2
%                 AUCFeatValues(1 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     0;
%             end
% 
%             AUCFeatValues(3 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     AUCFeatValues(1 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) + ...
%                     AUCFeatValues(2 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1));
% 
%             AUCFeatValues(4 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)) = ...
%                     sqrt(AUCFeatValues(1 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)).^2 + ...
%                     AUCFeatValues(2 + length(AUCIdxValues)*(j-1) + ...
%                     length(AUCBaseFeatNames)*(i-1)).^2);
% 
%         end
%     end
% 
% %     RefBaseline =  trapz(ibiTime, max(ibi, ibiBaseline)-ibiBaseline);
% %     AreaBelowRefBaseline = -trapz(ibiTime, min(ibi, ibiBaseline)-ibiBaseline);
% %     AreaIRefBaseline = sqrt(AreaAboveRefBaseline.^2 + AreaBelowRefBaseline.^2);
% %     AreaRefBaseline = AreaAboveRefBaseline + AreaBelowRefBaseline;
% %     
% % 
% %     AreaAboveRefBaseline =  trapz(ibiTime, max(ibi, ibiBaseline)-ibiBaseline);
% %     AreaBelowRefBaseline = -trapz(ibiTime, min(ibi, ibiBaseline)-ibiBaseline);
% %     AreaRefBaseline = AreaAboveRefBaseline + AreaBelowRefBaseline;
% % 
% %     AreaAboveRefMean =  trapz(ibiTime, max(ibi, mean(ibi))-mean(ibi));
% %     AreaBelowRefMean = -trapz(ibiTime, min(ibi, mean(ibi))-mean(ibi));
% %     AreaIRefMean = sqrt(AreaAboveRefMean.^2 + AreaBelowRefMean.^2);
% %     AreaRefMean = AreaAboveRefMean + AreaBelowRefMean;
% %     AreaAboveRefMed =  trapz(ibiTime, max(ibi, median(ibi))-median(ibi));
% %     AreaBelowRefMed = -trapz(ibiTime, min(ibi, median(ibi))-median(ibi));
% %     AreaIRefMed = sqrt(AreaAboveRefMed.^2 + AreaBelowRefMed.^2);
% %     AreaRefMed = AreaAboveRefMed + AreaBelowRefMed;
% % 
% 
    
        
    [~,newOrder] = ismember(possibleFeatNames,featNames);
    outputIdx = newOrder(newOrder~=0);
    outVec = possibleValues(outputIdx);

    if strcmp(output,"featvalues")
        varargout{1} = outVec;
    else
        varargout{1} = possibleValues;
        varargout{2} = possibleFeatNames;
    end
end
