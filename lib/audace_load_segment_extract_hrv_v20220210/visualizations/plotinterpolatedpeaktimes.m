pIdx = 1;
timesRefArp = [0,50];
[ecgRawData,ecgRawTime] = readecg(paths,pIdx,'output','raw','chanNum',[1,2],'times',timesRefArp);
%[arpRawData,arpRawTime] = readarp(paths,pIdx,'times',[0,30]);
%%
[ecgIbi,ecgIbiTime] = readecg(paths,pIdx,'output','ibi','times',timesRefArp);
%%
[ecgHb4AudioData,ecgHb4AudioTime] = ...
    readecg(paths,pIdx,'output','hbas1','times',timesRefArp);
ecgHb4AudioRelData = ecgHb4AudioData(ecgHb4AudioTime > 0);
ecgHb4AudioFs = round(1/median(diff(ecgHb4AudioTime)));
ecgHb4AudioFileName = string(paths.fileNameMatchTable.Audio_name(pIdx)) ...
    + "_ecg_hbas1.wav";
hbAs1DirName = "ecg_hbas1";
hbAs1DirPath = fullfile(paths.derivativesDirPath,hbAs1DirName);
if ~isfolder(hbAs1DirPath)
    mkdir(hbAs1DirPath);
end
ecgHb4AudioFilePath = fullfile(hbAs1DirPath,...
    ecgHb4AudioFileName);
audiowrite(ecgHb4AudioFilePath,ecgHb4AudioRelData,ecgHb4AudioFs);
%%
ecgHb4AudioTimeFromFs = ((1:length(ecgHb4AudioRelData))./ecgHb4AudioFs).';
ecgHbTimeFromHb4Audio = ecgHb4AudioTimeFromFs(ecgHb4AudioRelData>0);
ecgIbiFromHb4Audio = diff(ecgHbTimeFromHb4Audio)*1000;
ecgIbiTimeFromHb4Audio = ecgHbTimeFromHb4Audio(2:end);

succDiffIbiThresh = 1/1000;
succDiff = (abs(diff(ecgIbiTime) - ecgIbi(2:end)./1000) ...
        < succDiffIbiThresh);
%%
figure
plot(ecgIbiTime,60./(ecgIbi./1000))
hold on
plot(ecgIbiTimeFromHb4Audio,60./(ecgIbiFromHb4Audio./1000))

