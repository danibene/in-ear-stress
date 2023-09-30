function [interplIbi,interplIbiTime] = interplibi(ibi,ibiTime,desiredFs,interplMethod)
    xq = ((floor(min(ibiTime))*desiredFs):ceil(max(ibiTime))*desiredFs).'/desiredFs;
    maxSecsMissing = 10;
    tooMuchMissingIdx = (diff(ibiTime)>maxSecsMissing);
    interplIbiWithNan = interp1(ibiTime,ibi,xq,interplMethod);
    
    if max(tooMuchMissingIdx)>0
        warning("There are more than " + num2str(maxSecsMissing) + ...
            " seconds of missing data; " + ...
            "switching interpolation method to makima")
        possMissingIbiTimeIdx = 1:length(ibiTime);
        whereMissingEndIdx = possMissingIbiTimeIdx(tooMuchMissingIdx);
        whereMissingStartIdx = whereMissingEndIdx - 1;
        interplIbiWithNan4Missing = interp1(ibiTime,ibi,xq,'makima');
        whereMissingInterplTime = false(size(xq));
        for mIdx = 1:length(whereMissingStartIdx)
            whereMissingInterplTime(xq >= ibiTime(whereMissingStartIdx(mIdx)) ...
                & xq <= ibiTime(whereMissingEndIdx(mIdx))) = true;
        end
        interplIbiWithNan(whereMissingInterplTime) = ...
            interplIbiWithNan4Missing(whereMissingInterplTime);
    end
        
    toBeNan = (xq < min(ibiTime) | xq > max(ibiTime));
    interplIbiWithNan(toBeNan) = NaN;
    % remove NaNs at beginning and end of the 
    % interpolated IBI and time vectors
    notNanInd = ~isnan(interplIbiWithNan);
    interplIbi = interplIbiWithNan(notNanInd);
    interplIbiTime = xq(notNanInd);
end