%% set paths

% see if a folder called "data" is in the current directory
% if not keep going to the parent directory
% until the main directory containing all of the code and data is found
baseDirPath = pwd;
while ~isfolder(fullfile(baseDirPath,'data'))
    [baseDirPath,~,~] = fileparts(baseDirPath);
end
% add the code in the entire directory
addpath(genpath(baseDirPath))

% load the filepaths for existing data and for the code
paths = loadpaths(baseDirPath);
%%
pIdx = 11;

count = 0;
%%
for assumedFs=[500,501]
    for fileNum=1:3
        fileNameWithoutExt = strcat('hb_ext_iem_peaks_manual',num2str(fileNum));
        fileName = strcat(fileNameWithoutExt,'.txt');
        filePath = fullfile(paths.dataDirPath,'label_data',paths.fileNameMatchTable.Audio_name{pIdx},fileName);
        peaks = readtable(filePath);

        arpPeakTime = table2array(peaks(:,1));
        arpIbi = diff(arpPeakTime)*1000;
        arpIbiTime = arpPeakTime(2:end);
        possibleAbsDelay = 0.25;
        possibleTimes = [min(arpIbiTime) - possibleAbsDelay, max(arpIbiTime) + possibleAbsDelay];

        [iemlData,iemTime] = readarp(paths,pIdx,'sigName','IEML','times',possibleTimes);
        [iemrData,~] = readarp(paths,pIdx,'sigName','IEMR','times',possibleTimes);
        % using a zero-phase lowpass Butterworth filter 
        iemFs = 1/median(diff(iemTime));

        % fc = cutoff frequency in Hz
        fc = 50;

        % normalized cutoff frequency 
        Wn = fc/(iemFs/2);

        % order of filter
        orderButter = 4;

        % TF coefficients
        [b,a] = butter(orderButter, Wn, 'low');

        % zero-phase filtering
        filtIemlData = filtfilt(b,a,iemlData);
        filtIemrData = filtfilt(b,a,iemrData);
        [ecgCh1Data,ecgTime] = ...
            readecg(paths,pIdx,'output','raw','chanNum',1,...
            'times',possibleTimes,'assumedFs',assumedFs);
        [ecgCh2Data,~] = ...
            readecg(paths,pIdx,'output','raw','chanNum',2,...
            'times',possibleTimes,'assumedFs',assumedFs);

        %possibleTimes = [min(arpIbiTime) - 200, max(arpIbiTime) + 200];
        [ecgIbi,ecgIbiTime] = ...
            readecg(paths,pIdx,'output','ibi','times',possibleTimes,...
            'assumedFs',assumedFs);
        %[segNormIbi,segNormIbiTime,segAllIbi,segAllIbiTime] = readseg4ecg_test(dataPaths,pIdx,possibleTimes);
        ecgFs = 1/median(diff(ecgTime));

        possibleAbsDelay = 10;
        possibleTimes = [min(arpIbiTime) - possibleAbsDelay, max(arpIbiTime) + possibleAbsDelay];
        [refIbi,refIbiTime] = ...
            readecg(paths,pIdx,'output','ibi','times',possibleTimes,...
            'assumedFs',assumedFs);
        refBpm = 60./(refIbi./1000);
        refBpmTime = refIbiTime;
        bpm = 60./(arpIbi./1000);

        bpmTime = arpIbiTime;
        interplFs = 1000;
        interplBpmPosTimes = (floor(min(bpmTime))*interplFs:ceil(max(bpmTime))*interplFs).'/interplFs;
        interplBpmTime = interplBpmPosTimes(interplBpmPosTimes >= (min(bpmTime)) & interplBpmPosTimes <= max(bpmTime));
        interplBpm = interp1(bpmTime,bpm,interplBpmTime,'linear','extrap');    

        interplRefBpmPosTimes = (floor(min(refBpmTime))*interplFs:ceil(max(refBpmTime))*interplFs).'/interplFs;
        interplRefBpmTime = interplRefBpmPosTimes(interplRefBpmPosTimes >= min(refBpmTime) & interplRefBpmPosTimes <= max(refBpmTime));
        interplRefBpm = interp1(refBpmTime,refBpm,interplRefBpmTime,'linear','extrap');
        nDelays = length(interplRefBpm)-length(interplBpm);
        timeDelayVec = (1:nDelays)./interplFs - (min(interplBpmTime)-min(interplRefBpmTime));
        maeDelayVec = NaN(size(timeDelayVec));
        varManMinusRefTimes = NaN(size(timeDelayVec));
        for d=1:nDelays
            delInterplRefBpm = interplRefBpm(d:(length(interplBpm)+d-1));
            delInterplRefBpmTime = interplRefBpmTime(d:(length(interplBpm)+d-1));
            %delInterplRefIbi = (delInterplRefBpm./60)*1000;
            %peakTimeFirst = min(refBpmTime(refBpmTime>=min(delInterplRefBpmTime)));
            %[delRefBpmTime] = interplibi2peaktime(delInterplRefIbi,delInterplRefBpmTime,peakTimeFirst);
            absErr = abs(interplBpm - delInterplRefBpm);
            maeDelayVec(d) = mean(absErr);
            delRefBpmTime = refBpmTime(refBpmTime>=min(delInterplRefBpmTime) & refBpmTime <=max(delInterplRefBpmTime));
            timeArrLength = min(length(delRefBpmTime),length(bpmTime));
            manMinusRefTimes = delRefBpmTime(1:timeArrLength) - bpmTime(1:timeArrLength);
            varManMinusRefTimes(d) = var(manMinusRefTimes);
            %aMinusB = delRefBpmTime - bpmTime;
        end

        %iemlPeakHeight = prctile(iemlData,99);

        figDirPath = fullfile(baseDirPath,'figures','syncarpecg2021112');
        if ~isfolder(figDirPath)
            mkdir(figDirPath)
        end
        figure('units','normalized','outerposition',[0 0 1 1]);
        %tiledlayout(4,2)

        ax1 = subplot(4,2,1);
        plot(iemTime,iemlData)
        hold on
        plot(iemTime,filtIemlData)
        hold on
        scatter(arpIbiTime,iemlData(round(iemFs*(arpIbiTime-iemTime(1)))),'red','LineWidth',4)
        %scatter(arpIbiTime,ones(length(arpIbiTime),1)*iemlPeakHeight,'red','LineWidth',4)
        hold on
        scatter(ecgIbiTime,iemlData(round(iemFs*(ecgIbiTime-iemTime(1)))),'green','LineWidth',4)
        legend("IEML","Filtered IEML","Peaks manually detected from IEM","Peaks detected from ECG")
        xlabel("Time relative to start of IEM recording [seconds]")

        ax2 = subplot(4,2,3);
        plot(iemTime,iemrData)
        hold on
        plot(iemTime,filtIemrData)
        hold on
        scatter(arpIbiTime,iemrData(round(iemFs*(arpIbiTime-iemTime(1)))),'red','LineWidth',4)
        hold on
        scatter(ecgIbiTime,iemrData(round(iemFs*(ecgIbiTime-iemTime(1)))),'green','LineWidth',4)
        legend("IEMR","Filtered IEMR","Peaks manually detected from IEM","Peaks detected from ECG")
        xlabel("Time relative to start of IEM recording [seconds]")

        ax3 = subplot(4,2,5);
        plot(ecgTime,ecgCh1Data)
        hold on
        scatter(arpIbiTime,ecgCh1Data(round(ecgFs*(arpIbiTime-ecgTime(1)))),'red','LineWidth',4)
        hold on
        scatter(ecgIbiTime,ecgCh1Data(round(ecgFs*(ecgIbiTime-ecgTime(1)))),'green','LineWidth',4)
        legend("ECG Ch1","Peaks manually detected from IEM","Peaks detected from ECG")
        xlabel("Time relative to start of IEM recording [seconds]")

        ax4 = subplot(4,2,7);
        plot(ecgTime,ecgCh2Data)
        hold on
        scatter(arpIbiTime,ecgCh2Data(round(ecgFs*(arpIbiTime-ecgTime(1)))),'red','LineWidth',4)
        hold on
        scatter(ecgIbiTime,ecgCh2Data(round(ecgFs*(ecgIbiTime-ecgTime(1)))),'green','LineWidth',4)
        legend("ECG Ch2","Peaks manually detected from IEM","Peaks detected from ECG")
        xlabel("Time relative to start of IEM recording [seconds]")

        ax5 = subplot(4,2,2);
        if length(ecgIbiTime)  > length(arpIbiTime)
            plot(arpIbiTime,ecgIbiTime(1:length(arpIbiTime) - arpIbiTime))
        else
            plot(ecgIbiTime, ecgIbiTime - arpIbiTime(1:length(ecgIbiTime)));
        end
        legend("Delay between individual ECG and IEM peak times")
        xlabel("Time of IEM peak relative to start of IEM recording [seconds]")
        ylabel("Delay [seconds]")

        ax6 = subplot(4,2,4);
        plot(bpmTime,bpm)
        hold on
        plot(refIbiTime,refBpm)
        legend("BPM from peaks manually detected from IEM","BPM from peaks detected from ECG")
        xlabel("Time relative to start of IEM recording [seconds]")
        ylabel("Beats per minute")
        xlim([min(ecgIbiTime),max(ecgIbiTime)])

        ax7 = subplot(4,2,6);
        plot(timeDelayVec,maeDelayVec)
        hold on
        [~,I] = min(maeDelayVec);
        scatter(timeDelayVec(I),maeDelayVec(I))
        legend("Mean absolute error","Minimum")
        xlabel("Delay used for segment alignment: IEM segment start time - ECG segment start time [seconds]")
        ylabel("Mean absolute error [bpm]")

        ax8 = subplot(4,2,8);
        plot(timeDelayVec,varManMinusRefTimes)
        hold on
        [~,I] = min(varManMinusRefTimes);
        scatter(timeDelayVec(I),varManMinusRefTimes(I))
        legend("Variance of delay between individual ECG and IEM peak times","Minimum")
        xlabel("Delay used for segment alignment: IEM segment start time - ECG segment start time [seconds]")
        ylabel("Variability")

        linkaxes([ax1 ax2 ax3 ax4 ax5 ax6],'x')


        sgtitle(strcat("Participant 11: ",fileNameWithoutExt,...
            '_fs_',num2str(assumedFs)),'Interpreter','None')
        saveas(gcf,fullfile(figDirPath,strcat('sub11_',fileNameWithoutExt,...
            '_fs_',num2str(assumedFs),'.fig')))
        saveas(gcf,fullfile(figDirPath,strcat('sub11_',fileNameWithoutExt,...
            '_fs_',num2str(assumedFs),'.png')))
        %%
        count = count + 1;
        [~,I] = min(maeDelayVec);
        delayToAdd = timeDelayVec(I);
        delayEachSync(count) = delayToAdd;
        assumedFsEachSync(count) = assumedFs;
    end
end
