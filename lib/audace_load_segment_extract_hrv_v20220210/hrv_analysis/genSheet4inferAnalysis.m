%% set paths

% see if a folder called "data" is in the current directory
% if not keep going to the parent directory
% until the main directory containing all of the code and data is found
% (change path if data is in a different directory)
baseDirPath = pwd;
while ~isfolder(fullfile(baseDirPath,'data'))
    [baseDirPath,~,~] = fileparts(baseDirPath);
end
% add the code in the entire directory 
% (change path if code is in a different directory)
addpath(genpath(baseDirPath))

% load the file paths
paths = loadpaths(baseDirPath,'saveFile','saveNewFile');
resultsFileName = 'TimeLab_timeFeat_freqFeat_20220125.xlsx';
%% set constants
% for the minimum and maximum of the smoothed data, using a lowpass 
% filter with a cutoff frequency at 0.15 Hz
smoothFc = 0.15;
L = char(num2str(smoothFc));
L(L=='.')='p';
smoothedTimeFeatNames = strcat({'MinNN','MaxNN'},'_smoothed',L);
% set the time and frequency-domain HRV features to be extracted
timeFeatLabels = [{'MeanNN','SDNN','RMSSD','MinNN','MaxNN'},...
    smoothedTimeFeatNames];
freqFeatLabels = {'RatioLFHF', 'LF', 'HF', 'PeakLF', 'PeakHF'};
featLabels = [timeFeatLabels, freqFeatLabels];

% the uniform baseline time was defined as 180 seconds
expectedBaselineTime = 180;
% the tasks should have taken approximately the same time
expectedTskTimes = [300,300,180];
% the uniform anticipation time was defined as 60 seconds prior to the
% task, other than a few exceptions (accounted for later on in the script)
expectedAnticipationTime = 60;
% the uniform recovery time was defined as 60 seconds after the task, other
% than a few exceptions (also accounted for later on in the script)
expectedRecoveryTime = 60;
% if there is more than 50% of the segment's data missing then the entire
% segment is considered to be missing data
enoughDataPercSegLength = 50;
% in the case where multiple "sub-segments" are averaged for the segment
% if more than 50% of "sub-segments" are missing, then the entire segment
% is considered to be missing data as well
enoughDataPercNumSubSeg = 50;

% note: 30 seconds is considered an ultra short duration for time-domain
% HRV analysis; some features can be more reliable than others here
timeHrvSegSecs = 30;
% 120 seconds is also considered an ultra short duration for 
% frequency-domain HRV analysis
freqHrvSegSecs = 120;
% the hop size i.e. the shift in time between the start of subsequent 
% segments is 30 seconds for both 30 and 120-second segments
hopSecsAllSegs = 30;

tskNames = ["MENTAL","NOISE","CPT"];
% for each type of feature and condition, specify the duration of the
% individual segments 
restVsStressConds = ["BASELINE","TSK"];
segSecsMtx = zeros(length(restVsStressConds),length(featLabels));
% for the baseline the segments are 180 seconds (the entire baseline)
segSecsMtx(1,:) = expectedBaselineTime;
% for the tasks, the time-domain HRV features are computed over individual
% 30-second segments
segSecsMtx(2,1:length(timeFeatLabels)) = timeHrvSegSecs;
% and the frequency-domain HRV features are computed over 120-second
% segments
segSecsMtx(2,length(timeFeatLabels)+1 : ...
    length(timeFeatLabels)+length(freqFeatLabels)) = freqHrvSegSecs;
% for each type of feature and condition, specify the hop size 
hopSecsMtx = zeros(length(restVsStressConds),length(featLabels));
% for the baseline there's just a single 180-second segment
hopSecsMtx(1,:) = expectedBaselineTime;
% for the tasks, both 30- and 120-second length segments are computed every 
% 30 seconds
hopSecsMtx(2,1:length(timeFeatLabels)) = hopSecsAllSegs;
hopSecsMtx(2,length(timeFeatLabels)+1 : ...
    length(timeFeatLabels)+length(freqFeatLabels)) = hopSecsAllSegs;

% for each type of feature and condition, specify the duration of the
% individual "sub-segments" (only used for the baseline here) 

% copy the segment duration matrix, so that there is only one 
% sub-segment per task segment
subSegSecsMtx = segSecsMtx;
% in order for the features to be computed with the same segment durations
% regardlesss of stress/rest condition, change the sub-segment duration for 
% the baseline to match the segment duration for the task for the same 
% features 
subSegSecsMtx(1,1:length(timeFeatLabels)) = timeHrvSegSecs;
subSegSecsMtx(1,length(timeFeatLabels)+1 : ...
    length(timeFeatLabels)+length(freqFeatLabels)) = freqHrvSegSecs;
% copy the sub-segment duration matrix for the hop size matrix
subHopSecsMtx = subSegSecsMtx;
% change the values for the baseline such that the hop size is the same as
% the segments for the task data
subHopSecsMtx(1,length(timeFeatLabels)+1 : ...
    length(timeFeatLabels)+length(freqFeatLabels)) = hopSecsAllSegs;
% the values derived from the "sub-segments" will be averaged
% later to give a singular baseline value for the baseline segment

%% store the HRV features for all the segments

% initialize the table
varTypes = {'string','string','string','double','double','double','double','string'};
varNames = {'Participant','Task','Feature','SegmentStartRefTaskOnset','SegmentEndRefTaskOnset','Value','segDur','Time'};
resultsTable = table('Size',[0 length(varNames)],'VariableTypes',varTypes,'VariableNames',varNames);

count = 0;
% for each participant except for 27, who was excluded from this analysis
for pIdx=1:length(paths.fileNameMatchTable.Audio_name)   
    if ~contains(string(paths.fileNameMatchTable.Audio_name(pIdx)),'27_H-PAB')
        % load the table with the timestamps of the baseline and tasks
        % (segmented such to have uniform duration when possible)
        timestampsTable = loadtsktimes(paths,pIdx);
        % load all ECG data collected after the ARP started recording
        [allIbi,allIbiTime] = readecg(paths,pIdx,[0,Inf]);
        % for each of the 3 tasks, extract the timestamps of the baseline
        % preceding the task, the anticipation period, the task itself, and
        % the recovery period
        for tIdx=1:length(tskNames)
            baselineIdx = strcmp(table2array(timestampsTable(:,3)),...
                        strcat("BASELINEPRE_",tskNames(tIdx)));
            baselineTimestamps = ...
                table2array(timestampsTable(baselineIdx,1:2));
            anticipationIdx = strcmp(table2array(timestampsTable(:,3)),...
                        strcat(tskNames(tIdx),"_ANTICIPATION"));
            anticipationTimestamps = ...
                table2array(timestampsTable(anticipationIdx,1:2));
            tskIdx = strcmp(table2array(timestampsTable(:,3)),...
                strcat(tskNames(tIdx),"_TSK"));
            tskTimestamps = ...
                table2array(timestampsTable(tskIdx,1:2));
            recoveryIdx = strcmp(table2array(timestampsTable(:,3)),...
                strcat(tskNames(tIdx),"_RECOVERY"));
            recoveryTimestamps = ...
                table2array(timestampsTable(recoveryIdx,1:2));
            % estimate the approximate real time of the task with the
            % difference of the task timestamps, since some participants
            % unexpectedly had to stop the task and so the time after that
            % should be considered missing data
            approxRealTskDur = diff(tskTimestamps);
            % check how the anticipation should be included, if at all:
            % for participants whose task anticipation should be completely
            % discarded because of an unexpected event, their timestamps 
            % were replaced by NaNs in loadtsktimes.m
            if max(isnan(anticipationTimestamps))
                includeAnticipation = 'excluded';
            else
                % if the anticipation period happened sometime that was
                % not directly before the start of the task, segments that
                % consist of overlapping anticipation & task data should be
                % consdiered missing
                if anticipationTimestamps(2) ~= tskTimestamps(1)
                    includeAnticipation = 'discontinuousTask';
                else
                    % otherwise the 60 seconds before the task are taken
                    includeAnticipation = 'expected';
                end
            end
            % the task length in most cases was approximately 300 or 180
            % seconds, for mental&noise or CPT respectively
            expectedTskTime = expectedTskTimes(tIdx);
            
            % for each HRV feature
            for fIdx=1:length(featLabels)
                % for the baseline and task conditions
                for rest=1:length(restVsStressConds)
                    % get the segment and hop size based on the current 
                    % baseline/task condition and HRV feature
                    segSecs = segSecsMtx(rest,fIdx);
                    hopSecs = hopSecsMtx(rest,fIdx);
                    
                    if rest==1
                        % for each baseline, 180 seconds of data should be
                        % analyzed 
                        expectedAllPhaseDuration = expectedBaselineTime;
                        newTskLabel = strcat("BASELINEPRE_",...
                            tskNames(tIdx),"_TSK");
                    else
                        % for each task, the data used for the segments 
                        % in most cases will be 60 seconds of anticipation, 
                        % the duration of the task, and 60 seconds of
                        % recovery
                        expectedAllPhaseDuration = expectedAnticipationTime ...
                            + expectedTskTime + expectedRecoveryTime;
                        newTskLabel = strcat(tskNames(tIdx),"_TSK");
                    end
                    % determine the expected number of segments given the
                    % duration of each segment, the hop size, and the
                    % duration of the data available for the current 
                    % condition in most cases 
                    expectedNSegs = floor((expectedAllPhaseDuration - ...
                        segSecs)/hopSecs + 0.00001)+1;
                    possSegIdx = 1:expectedNSegs;
                    if rest==1
                        % each baseline will only have one value per
                        % feature so the start and end of this single 
                        % segment is simply the beginning and end of the 
                        % baseline
                        t1 = baselineTimestamps(1);
                        tEnd = baselineTimestamps(2);
                        % baseline segment labeled as t00 in results table
                        timeLabel = "t00";
                    else
                        % for the tasks, the starting time of the first 
                        % segment should be 60 seconds before the start of
                        % the task in most cases
                        expectedSegT1RefTskStart = ...
                            -1*expectedAnticipationTime;
                        % the starting time of all the segments can be
                        % determined with the hop size and number of
                        % segments
                        expectedSegT1RefTskStart = ...
                            expectedSegT1RefTskStart + ...
                            hopSecs*(possSegIdx-1);
                        % the end time of all the segments is the segment 
                        % duration added to each segment's starting time 
                        expectedSegTEndRefTskStart = ...
                            expectedSegT1RefTskStart + segSecs;
                        % some segments could contain data from both the
                        % anticipation and task, in which case the data 
                        % should be continuous 
                        overlappingAnticipationTskIdx = ...
                            (expectedSegT1RefTskStart < 0 & ...
                            expectedSegTEndRefTskStart > 0);
                        % depending on the segment length, some segments
                        % could contain data only from the anticipation
                        % period and not overlapping with the task, in 
                        % which case it's not problematic the include these
                        % segments if there's discontinuity 
                        
                        nonOverlappingAnticipationIdx = ...
                            (expectedSegT1RefTskStart < 0 & ...
                            expectedSegTEndRefTskStart <= 0);                        
                        % some segments could contain data from both the
                        % task and recovery, in which case the data 
                        % should be continuous - if the task ended earlier 
                        % than expected, that's considered missing data
                        overlappingTskRecoveryIdx = ...
                            (expectedSegT1RefTskStart < expectedTskTime & ...
                            expectedSegTEndRefTskStart > expectedTskTime);
                        % discontinuous recovery data can be used in
                        % case of segments with only recovery data
                        nonOverlappingTskRecoveryIdx = ...
                            (expectedSegT1RefTskStart >= expectedTskTime & ...
                            expectedSegTEndRefTskStart > expectedTskTime);
                        
                        % if the task ended earlier than it was expected
                        % to, there could be segments that should be marked
                        % as missing data after the task ended but before
                        % the task was expected to end
                        tskEndedIdx = (expectedSegTEndRefTskStart < ...
                            expectedTskTime & expectedSegTEndRefTskStart > ...
                            approxRealTskDur);
                        
                        % each time label, including anticipation and 
                        % recovery, is labeled in chronological order from
                        % t01 to tN 
                        timeLabel = string(zeros(1,expectedNSegs));
                        for sIdx=possSegIdx
                            if sIdx>9
                                timeLabel(sIdx) = strcat('t',num2str(sIdx,'%d'));
                            else
                                timeLabel(sIdx) = strcat('t0',num2str(sIdx,'%d'));
                            end
                        end
                    end
                    % create a row in the results table for each segment
                    for sIdx=possSegIdx
                        count = count + 1;
                        % participant ID
                        resultsTable.Participant(count) = ...
                            paths.fileNameMatchTable.Audio_name(pIdx);
                        % task label
                        resultsTable.Task(count) = newTskLabel;
                        % HRV feature name
                        resultsTable.Feature(count) = featLabels{fIdx};
                        % time label (t00,t01,...)
                        resultsTable.Time(count) = timeLabel(sIdx);
                        % segment duration in seconds
                        resultsTable.segDur(count) = segSecs;
                        if rest == 1
                            % only one segment for the baseline recording
                            % so the start of the segment is the start of
                            % the baseline
                            segT1 = t1;
                            % several conditions for the segment being 
                            % marked as missing are checked:
                            %
                            % - if the anticipation supposed to be 
                            % excluded and the current segment should 
                            % include anticipation data
                            %
                            % - if the anticipation and task are not
                            % continuous and the current segment should
                            % include both anticipation and task data
                            %
                            % - if the task ended early and the current
                            % segment should include both task and
                            % recovery data
                            %
                            % - if the task ended early and the current
                            % segment should include data after the task
                            % ended
                        elseif (strcmp(includeAnticipation,'excluded') && ...
                                (ismember(sIdx,...
                                possSegIdx(nonOverlappingAnticipationIdx)) || ...
                                ismember(sIdx,...
                                possSegIdx(overlappingAnticipationTskIdx)))) ...
                                || ...
                                (strcmp(includeAnticipation,...
                                'discontinousTask') && ...
                                ismember(sIdx,...
                                possSegIdx(overlappingAnticipationTskIdx))) ...
                                || ...
                                (approxRealTskDur < expectedTskTime && ...
                                ismember(sIdx,...
                                possSegIdx(overlappingTskRecoveryIdx))) ...
                                || ...
                                ismember(sIdx,possSegIdx(tskEndedIdx))
                            
                            resultsTable.SegmentStartRefTaskOnset(count) = ...
                                NaN;
                            resultsTable.SegmentEndRefTaskOnset(count) = ...
                                NaN;
                            resultsTable.Value(count) = NaN;
                            
                        else
                            % if the current segment should contain only 
                            % anticipation data, the start
                            % time of the current segment is determined  
                            % with the anticipation timestamps rather than 
                            % only relative to the task timestamps
                            if ismember(sIdx,...
                                possSegIdx(nonOverlappingAnticipationIdx))
                            
                                segT1RefAnticipationStart = ...
                                    expectedSegT1RefTskStart(sIdx) - ...
                                min(expectedSegT1RefTskStart(possSegIdx(nonOverlappingAnticipationIdx)));
                                
                                segT1 = anticipationTimestamps(1) + segT1RefAnticipationStart;
                            % if the current segment should contain only 
                            % recovery data, the start time of the current
                            % segment is determined with the recovery 
                            % timestamps rather than only relative to the 
                            % task timestamps
                            elseif ismember(sIdx,...
                                    possSegIdx(nonOverlappingTskRecoveryIdx))
                            
                                segT1RefRecoveryStart = ...
                                    expectedSegT1RefTskStart(sIdx) - ...
                                min(expectedSegT1RefTskStart(possSegIdx(nonOverlappingTskRecoveryIdx)));
                            
                                segT1 = recoveryTimestamps(1) + ...
                                    segT1RefRecoveryStart;
                                
                            else
                                % otherwise the segment start time is based 
                                % on the task timestamps
                                segT1 = tskTimestamps(1) + expectedSegT1RefTskStart(sIdx);
                            end
                        end
                        % if the segment has not been excluded, extract the
                        % HRV feature from the interbeat intervals within 
                        % the segment timestamps 
                        if ~isnan(resultsTable.Value(count))
                            segTEnd = segT1 + segSecs;
                            % include the segment timestamps relative to 
                            % the start of the task in the table
                            resultsTable.SegmentStartRefTaskOnset(count) = ...
                                segT1 - tskTimestamps(1);
                            resultsTable.SegmentEndRefTaskOnset(count) = ...
                                segTEnd - tskTimestamps(1);
                            % use the interbeat intervals within the
                            % segment timestamps
                            segIbi = allIbi(allIbiTime > segT1 & ...
                                allIbiTime <= segTEnd);
                            segIbiTime = allIbiTime(allIbiTime > ...
                                segT1 & allIbiTime <= segTEnd);
                            
                            % the baseline segment is further divided into
                            % "sub-segments" 
                            subSegSecs = subSegSecsMtx(rest,fIdx);
                            subHopSecs = subHopSecsMtx(rest,fIdx);
                            nSubSegs = floor(abs((((-1)*segT1 + segTEnd) - ...
                                (subSegSecs))/subHopSecs))+1;
                            subSegResults = zeros(1,nSubSegs);
                            
                            % (for the task segments there should be only
                            % one sub-segment per segment)
                            for subSegIdx=1:nSubSegs
                                
                                subSegT1 = segT1 + subHopSecs*(subSegIdx-1);
                                subSegTEnd = subSegT1 + subSegSecs;
                                subSegIbi = allIbi(allIbiTime > ...
                                    subSegT1 & allIbiTime <= subSegTEnd);
                                subSegIbiTime = allIbiTime(allIbiTime > ...
                                    subSegT1 & allIbiTime <= subSegTEnd);
                                enoughData = 1;
                                % if the sum of the interbeat intervals is
                                % less than half of the duration of the
                                % (sub-)segment, it is considered missing
                                % data, since without missing data, 
                                % the interbeat intervals in a segment 
                                % should add up to approximately the
                                % segment length
                                if sum(subSegIbi)/1000 < ...
                                        (enoughDataPercSegLength/100)...
                                        *(subSegTEnd-subSegT1)
                                    enoughData=0;
                                end
                                if enoughData==1
                                    % if there is enough data extract the
                                    % current HRV feature 
                                      subSegResults(subSegIdx) = ...
                                          extracthrvfeat(subSegIbi,...
                                          subSegIbiTime,...
                                          'featNames',{featLabels{fIdx}},...
                                          'smoothFc',smoothFc);
                                else
                                    subSegResults(subSegIdx) = NaN;
                                end
                            end
                            % if too many sub-segments were considered
                            % missing data, the entire segment is
                            % considered missing
                            if length(subSegResults(~isnan(subSegResults))) < ...
                                        (nSubSegs*(enoughDataPercNumSubSeg/100))
                                resultsTable.Value(count) = NaN;
                            else
                                % otherwise take the mean of the
                                % sub-segment values to get a single value
                                % for the segment 
                                resultsTable.Value(count) = ...
                                    mean(subSegResults,'omitnan');
                            end
                        end
                    end
                end
            end
        end
    end
end

%% save the spreadsheet
if ~isfolder(paths.hrvFeatDirPath)
    mkdir(paths.hrvFeatDirPath)
end
writetable(resultsTable,fullfile(paths.hrvFeatDirPath,resultsFileName));

