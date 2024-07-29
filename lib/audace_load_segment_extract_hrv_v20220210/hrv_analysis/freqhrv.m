function [outVec] = freqhrv(featNames,ibi,ibiTime)
    interplFs = 2;
    interplMethod = 'spline';
    [interplIbi,interplIbiTime] = ...
            interplibi(ibi,ibiTime,...
            interplFs,interplMethod);
    winSamp = 256;
    overlapSamp = (31/32)*256;
    [pxx, f, ~] = psd4ibi(interplIbi,interplIbiTime,winSamp,overlapSamp);
    
    fMinULF = 0;
    fMaxULF = 0.0033;
    
    fMinVLF = 0.0033;
    fMaxVLF = 0.04;
    
    fMinLF = 0.04;
    fMaxLF = 0.15;
    
    fMinHF = 0.15;
    fMaxHF = 0.4;
    
    fMinVHF = 0.4;
    fMaxVHF = 0.5;
    
    fMinUHF = 0.5;
    fMaxUHF = 1;
    
    ULF = bandpower(pxx,f,[fMinULF fMaxULF],'psd');
    VLF = bandpower(pxx,f,[fMinVLF fMaxVLF],'psd');
    LF = bandpower(pxx,f,[fMinLF fMaxLF],'psd');
    HF = bandpower(pxx,f,[fMinHF fMaxHF],'psd');
    VHF = bandpower(pxx,f,[fMinVHF fMaxVHF],'psd');
    UHF = bandpower(pxx,f,[fMinUHF fMaxUHF],'psd');
    
    ratioLFHF = LF/HF;
    
    idxLF = (f >= fMinLF & f <= fMaxLF);
    fLF = f(idxLF);
    [~, idxPeakLF] = max(pxx(idxLF));
    peakLF = fLF(idxPeakLF);
    
    idxHF = (f >= fMinHF & f <= fMaxHF);
    fHF = f(idxHF);
    [~, idxPeakHF] = max(pxx(idxHF));
    peakHF = fHF(idxPeakHF);
    
    possibleFeatNames = {'ULF','VLF','LF','HF','VHF','UHF'...
                'RatioLFHF','PeakLF','PeakHF'};
    possibleValues = [ULF,VLF,LF,HF,VHF,UHF,...
        ratioLFHF,peakLF,peakHF];
    [~,newOrder] = ismember(possibleFeatNames,featNames);
    possIdx = 1:length(possibleValues);
    outputIdx = possIdx(newOrder~=0);
    outVec = possibleValues(outputIdx);
end