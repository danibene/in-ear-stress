function [peakTime] = interplibi2peaktime(interplIbi,interplIbiTime,peakTimeFirst)
    count = 1;
    peakTime = [];
    peakTime(count) = peakTimeFirst;
    minBpm = 40;
    maxTimeBetweenHb = 60/minBpm;
    while max(peakTime) <= max(interplIbiTime)
        closeInterplIbiTimeIdx = (interplIbiTime > peakTime(count) & ...
            interplIbiTime < peakTime(count) + maxTimeBetweenHb*3);
        closeInterplIbi = interplIbi(closeInterplIbiTimeIdx);
        possIbiTimes = peakTime(count) + closeInterplIbi./1000;
        [~,idx] = min(abs(possIbiTimes-interplIbiTime(closeInterplIbiTimeIdx)));
        currIbi = closeInterplIbi(idx);
        count = count + 1;
        peakTime(count) = peakTime(count-1) + currIbi/1000;
    end
    peakTime = ensurecolumn(peakTime(1:end-1));
end