function [outVec] = nonlinearhrv(featNames,ibi,ibiTime,timeMissing,ibiBaseline)
    
    x = ibi(1:end-1);
    y = ibi(2:end);

    yTime = ibiTime(2:end);
    yOnlySucc = y;
    yOnlySucc(ismember(yTime,timeMissing)) = NaN;
    
    succDiffWithNan = yOnlySucc - x;
    succDiff = succDiffWithNan(~isnan(succDiffWithNan));
    
    lenDiff = length(succDiff);
    
    x1 = (x - y) / sqrt(2);
    x2 = (x + y) / sqrt(2);
    
    SD1 = std(x1);
    SD2 = std(x2);
    
    % SD1 / SD2
    SD1SD2 = SD1/SD2;

    % Area of ellipse described by SD1 and SD2
    S = pi * SD1 * SD2;

    % CSI / CVI
    T = 4 * SD1;
    L = 4 * SD2;
    CSI = L / T;
    CVI = log10(L * T);
    CSI_Modified = L * 2 / T;    
    
    decIdx = (succDiffWithNan > 0);
    decSuccDiff = succDiffWithNan(decIdx); % set of points above IL where y > x
    accIdx = (succDiffWithNan < 0);
    accSuccDiff = succDiffWithNan(accIdx);  % set of points below IL where y < x
    noChangeIdx = (succDiffWithNan == 0);
    noChangeSuccDiff = succDiffWithNan(noChangeIdx);

    % Distances to centroid line l2
    centroidX = mean(x);
    centroidY = mean(y);
    distCentroidAll = abs((x - centroidX) + (y - centroidY)) / sqrt(2);
    distBaselineAll = abs((x - ibiBaseline) + (y - ibiBaseline)) / sqrt(2);

    % Distances to LI
    distAll = abs(y - x) / sqrt(2);

    % Calculate the angles
    thetaAll = abs(atan(1) - atan(y ./ x));  % phase angle LI - phase angle of i-th point
    % Calculate the radius
    r = sqrt(x.^2 + y.^2);
    % Sector areas
    sAll = 1 / 2 * thetaAll .* r.^2;

    % Guzik's Index (GI)
    denGI = sum(distAll);
    numGI = sum(distAll(decIdx));
    GI = (numGI / denGI) * 100;

    % Slope Index (SI)
    denSI = sum(thetaAll);
    numSI = sum(thetaAll(decIdx));
    SI = (numSI / denSI) * 100;

    % Area Index (AI)
    denAI = sum(sAll);
    numAI = sum(sAll(decIdx));
    AI = (numAI / denAI) * 100;

    % Porta's Index (PI)
    m = lenDiff - length(noChangeSuccDiff);  % all points except those on LI
    b = length(accSuccDiff);  % number of points below LI
    PI = (b / m) * 100;

    % Short-term asymmetry (SD1)
    SD1d = sqrt(sum(distAll(decIdx).^2) / (lenDiff - 1));
    SD1a = sqrt(sum(distAll(accIdx).^2) / (lenDiff - 1));

    SD1I = sqrt(SD1d.^ 2 + SD1a.^2);
    C1d = (SD1d / SD1I).^2;
    C1a = (SD1a / SD1I).^2;

    % Long-term asymmetry (SD2)
    longtermDec = sum(distCentroidAll(decIdx).^2) / (lenDiff - 1);
    longtermAcc = sum(distCentroidAll(accIdx).^2) / (lenDiff - 1);
    longtermNoChange = sum(distCentroidAll(noChangeIdx).^2) / (lenDiff - 1);
    
    % Long-term asymmetry (SD2) ab
    longtermDecRefBaseline = sum(distBaselineAll(decIdx).^2) / (lenDiff - 1);
    longtermAccRefBaseline = sum(distBaselineAll(accIdx).^2) / (lenDiff - 1);
    longtermNoChangeAb = sum(distBaselineAll(noChangeIdx).^2) / (lenDiff - 1);

    SD2d = sqrt(longtermDec + 0.5 .* longtermNoChange);
    SD2a = sqrt(longtermAcc + 0.5 .* longtermNoChange);
    
    SD2dRefBaseline = sqrt(longtermDecRefBaseline + 0.5 .* longtermNoChangeAb);
    SD2aRefBaseline = sqrt(longtermAccRefBaseline + 0.5 .* longtermNoChangeAb);

    SD2I = sqrt(SD2d.^2 + SD2a.^2);
    SD2IRefBaseline = sqrt(SD2dRefBaseline.^2 + SD2aRefBaseline.^2);

    C2d = (SD2d / SD2I).^2;
    C2a = (SD2a / SD2I).^2;
    
    % Total asymmetry (SDNN)
    SDNNd = sqrt(0.5 * (SD1d.^2 + SD2d.^ 2));  % SDNN deceleration
    SDNNa = sqrt(0.5 * (SD1a.^2 + SD2a.^ 2));  % SDNN acceleration
    SDNN_nonlinear = sqrt(SDNNd.^2 + SDNNa.^2);  % should be similar to SDNN
    Cd = (SDNNd / SDNN_nonlinear).^2;
    Ca = (SDNNd / SDNN_nonlinear).^2;

    %CorDim = correlationDimension(ibi);
    CorDim = NaN;
    %ApEn = approximateEntropy(ibi);
    ApEn = NaN;
    
    possibleFeatNames = {'SD1', 'SD2', 'SD1SD2', 'SD2d','SD2a',...
        'SD2dRefBaseline','SD2aRefBaseline',...
        'SD2IRefBaseline',...
        'SD2I','C2d','C2a',...
        'SDNNd','SDNNa','SDNN_nonlinear','Cd','Ca',...
        'GI','SI','AI','PI','C1d','C1a',...
        'CorDim','ApEn'};
    possibleValues = [SD1, SD2, SD1SD2, SD2d,SD2a,...
        SD2dRefBaseline,SD2aRefBaseline,...
        SD2IRefBaseline,...
        SD2I,C2d,C2a,...
        SDNNd,SDNNa,SDNN_nonlinear,Cd,Ca,...
        GI,SI,AI,PI,C1d,C1a,...
        CorDim,ApEn];
    [~,newOrder] = ismember(possibleFeatNames,featNames);
    possIdx = 1:length(possibleValues);
    outputIdx = possIdx(newOrder~=0);
    outVec = possibleValues(outputIdx);
end