function [smoothedIbi,interplIbiTime] = smoothibi(ibi,ibiTime,smoothFc)

    interplFs = 2;
    interplMethod = 'spline';
    [interplIbi,interplIbiTime] = ...
            interplibi(ibi,ibiTime,...
            interplFs,interplMethod);

    % using a zero-phase lowpass Butterworth filter 
    
    % smoothFc = cutoff frequency in Hz
    
    % normalized cutoff frequency 
    Wn = smoothFc/(interplFs/2);

    % order of filter
    orderButter = 2;

    % TF coefficients
    [b,a] = butter(orderButter, Wn, 'low');

    % zero-phase filtering
    smoothedIbi = filtfilt(b,a,interplIbi);
end