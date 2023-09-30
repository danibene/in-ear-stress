function [meanPxx, f, winsPxx] = psd4ibi(ibi,ibiTime,winSamp,overlapSamp)
% Detrends and computes power spectrum from interbeat interval vector
%
% IN	ibi                     interbeat intervals (ms)
%		ibiTime                 time vector corresponding to ibi
%       winSamp                 number of samples in each window
%       overlapSamp             number of samples overlapping in each window
%
%
% OUT	pxx                     Power vector averaged over windows
%		f                       Frequencies corresponding to power
%		winsPxx              	Power vector at each window
    
    % compute the sampling frequency
    fs = round(1/median(diff(ibiTime)));
    % create indices for windows
    if length(ibiTime) < winSamp
        nZeros = winSamp - length(ibiTime);
        wins = [ensurecolumn(1:length(ibiTime));zeros(nZeros,1)];
    else
        wins = buffer(1:length(ibiTime),winSamp,overlapSamp,'nodelay');
    end
    % number of windows
    nWins = size(wins,2);
    % the size of the FFT is the same as the window size
    fftSamp = winSamp;
    % the frequencies corresponding to power
    f = ensurecolumn(fs/2*linspace(0,1,fftSamp/2+1));  
    % each window will be multiplied by a hamming window function
    win = ensurecolumn(hamming(winSamp));
    % create empty vector for summing power over windows
    totalPxx = zeros(length(f),1);
    % create empty vector for power at each window
    winsPxx = zeros(nWins,fftSamp/2+1);
    
    % loop over windows
    for currWin = 1:nWins
        % take the window indices (towards the end there might be zeros)
        winIdxWithZeros = wins(:,currWin);
        % remove any zeros
        currWinIdx = winIdxWithZeros(winIdxWithZeros~=0);
        % and check how many were removed, if any
        nZeros = length(winIdxWithZeros) - length(currWinIdx);
        
        if nZeros == 0
            % if there aren't any zeros, the segment used for the window 
            % is the IBI vector at the specified indices 
            winIbi = ensurecolumn(detrend(ibi(currWinIdx),3));
        else
            % if there are, then pad the rest of the segment with zeros so
            % that the size matches the other windows
            winIbi = [ensurecolumn(detrend(ibi(currWinIdx),3));zeros(nZeros,1)];
        end
        
        % apply the windowing function
        %windowedIbi = winIbi.*win;
        % apply the FFT and divide results by the FFT size
        %Y = fft(windowedIbi,fftSamp)/fftSamp;
        % select one side of the spectrum and account for loss of energy 
        % due to windowing function
        [winPxx,f] = periodogram(winIbi,win,[],fs);
        % add power from current window to the total power vector
        totalPxx = totalPxx + ensurecolumn(winPxx);
        % save power from current window in matrix
        winsPxx(currWin,:) = ensurecolumn(winPxx);
    end
    
    % average power
    meanPxx = totalPxx ./ nWins;
end