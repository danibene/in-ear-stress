function [hbVector] = baselineEcgPeak(ECG,fs)
% from lib\CritiasStress\syncing_segmenting\segmentingScript.m
    len = Inf;
    len = min(len,length(ECG));
    MIN_HB_PKS_DIST = 1/(200/60) * fs;
    [PKS,LOCS] = findpeaks(ECG,'MinPeakDistance',MIN_HB_PKS_DIST,'MinPeakProminence',0.5);
    hbVector = zeros(size(ECG));
    hbVector(round(LOCS)) = 1;
end