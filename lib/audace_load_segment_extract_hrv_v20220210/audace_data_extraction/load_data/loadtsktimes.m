function [varargout] = loadtsktimes(paths,pIdx,varargin)
% Find timestamps of task segments with uniform duration
%
% IN		paths                   struct containing file paths
%           pIdx                    numerical participant ID
%			saveFile                option to save new file with timestamps
%           output                  option to specify output
%
% OUT		varargout               default only table with unified task
%                                   segments, but can also include original
%                                   task timestamps
%
% v20211213 DB
    
    % parse inputs
    defaultSaveFile = 'saveNewFile';
    defaultoutput = 'uniformLength';
    expectedSaveFile = {'readFromFile','saveNewFile','loadWithoutSave'};
    expectedOutput = {'uniformLength','originalLength','diffUniformOriginal'};

    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    addRequired(p,'paths');
    addRequired(p,'pIdx',validScalarPosNum);
    addParameter(p,'saveFile',defaultSaveFile,...
                 @(x) any(validatestring(x,expectedSaveFile)));
    addParameter(p,'output',defaultoutput,...
        @(x) any(validatestring(x,expectedOutput)));
    
    parse(p,paths,pIdx,varargin{:});
    paths = p.Results.paths; 
    pIdx = p.Results.pIdx;
    saveFile = p.Results.saveFile;
    output = p.Results.output;
    
    % read labels with real task, baseline, and preparation times
    realTimeTableFilePath = fullfile(paths.arpMkr4tskSegDirPath,...
        strcat(paths.fileNameMatchTable.Audio_name(pIdx),...
        paths.arpMkr4tskSegFileSuffix));
    realTimeTable = readtable(realTimeTableFilePath);
    realTimeTable.Properties.VariableNames = ....
        {'LabelStartTimeRefArpStart', 'LabelEndTimeRefArpStart','Label'};
    
    uniTimeTableFilePath = fullfile(paths.tskTimestampsDirPath,...
        strcat(paths.fileNameMatchTable.Audio_name(pIdx),...
        paths.tskTimestampsFileSuffix));
    
    if strcmp(saveFile,"readFromFile") && ...
            exist(uniTimeTableFilePath,'file')
        uniTimeTable = readtable(uniTimeTableFilePath);
    else

        if ~isfolder(paths.tskTimestampsDirPath)
            mkdir(paths.tskTimestampsDirPath)
        end
        %%
        fileList = dir(fullfile(paths.arpMkr4tskSegDirPath,...
        strcat('*',paths.arpMkr4tskSegFileSuffix)));
        %% set constants
        % mental arithmetic, noise, and cold pressor test
        tskNames = ["MENTAL","NOISE","CPT"];
        % the intended duration of each of the respective tasks in seconds
        intendedTskTimes = [300,300,180];

        % 4 phases per task: baseline, anticipation, task and recovery
        numPhaseTsk = 4;
        newLabelNames = string(zeros(length(tskNames)*numPhaseTsk,1));
        for tsk=1:length(tskNames)
        newLabelNames((tsk-1)*numPhaseTsk+1) = ...
            strcat("BASELINEPRE_",tskNames(tsk));
        newLabelNames((tsk-1)*numPhaseTsk+2) = ...
            strcat(tskNames(tsk),"_ANTICIPATION");
        newLabelNames((tsk-1)*numPhaseTsk+3) = ...
            strcat(tskNames(tsk),"_TSK");
        newLabelNames((tsk-1)*numPhaseTsk+4) = ...
            strcat(tskNames(tsk),"_RECOVERY");
        end
        % 60 seconds before the start of the task taken as the anticipation phase
        % (task preparation time was variable but 60 seconds chosen for uniform
        % length across participants)
        uniAnticipationTime = 60;
        % 60 seconds after the end of task phase taken as the recovery phase
        uniRecoveryTime = 60;
        % 180 seconds in the middle of the actual baseline recording taken as the
        % baseline phase
        uniBaselineTime = 180;

        % the threshold at which the absolute difference between the 
        % real and intended task duration (in seconds) is considered too long 
        % and the duration considered for the analysis will be different
        threshDiffIntendedRealTskTime = 16;
        % the threshold at which the absolute difference between the 
        % task preparation end and the beginning of the task is considered too long 
        % and the anticipation considered for the analysis will be different
        threshDiffIntendedRealAnticipation = 90;

        % in the cases where the segment duration is different from the
        % intended duration, the time which the new duration should be divisible by
        % (30 because 30-second segments were taken for the time-domain features)
        divisibleTime = 30;

        anticipationInvalidSubs = ["01_F-HG","11_F-LV"];
        anticipationInvalidTasks = ["NOISE","NOISE"];

        % initialize table to save new segments with uniform duration
        varTypes = {'double','double','cell'};
        uniTimeTable = table('Size',[0 3],'VariableTypes',varTypes);

        realTskTime = NaN(1,length(tskNames));
        realBaselineTime = NaN(1,length(tskNames));
        diffIntendedRealAnticipation = NaN(1,length(tskNames));
        % for each task
        for tsk=1:length(tskNames)
            % get the index of the row with the timestamps of the "preparation" 
            % for the baseline preceding the current task, i.e. the participant 
            % being instructed that the baseline recording is starting
            baselinePrepIdx = strcmp(realTimeTable.Label,...
                strcat("BASELINEPRE_",tskNames(tsk),"_PREP"));
            % the real baseline starts at the end of the baseline preparation
            startTimeBaseline = realTimeTable.LabelEndTimeRefArpStart(baselinePrepIdx);
            % use the beginning of the audio recording as the start of the 
            % baseline if the baseline start time is missing and it is the 
            % baseline preceding the first task (for participant 20 the order
            % of the first two tasks were reversed) since the preparation was 
            % probably right before starting audio recording
            if isempty(startTimeBaseline) && (tsk==1 || (tsk==2 && ...
                    strcmp(extractBefore(fileList(20).name,...
                    paths.arpMkr4tskSegFileSuffix),'20_H-GL')))
                startTimeBaseline = 0;
            end

            % get the index of the row with the timestamps of the preparation for the task
            tskPrepIdx = strcmp(realTimeTable.Label,strcat(tskNames(tsk),"_PREP"));
            endTimeTskPrep = realTimeTable.LabelEndTimeRefArpStart(tskPrepIdx);
            % the real baseline ends at the start of the task preparation
            endTimeBaseline = realTimeTable.LabelStartTimeRefArpStart(tskPrepIdx);
            % get the index of the row with the timestamps of the task
            tskIdx = strcmp(realTimeTable.Label,strcat(tskNames(tsk),"_TSK"));
            % get the timestamp of when the task started
            startTimeTsk = realTimeTable.LabelStartTimeRefArpStart(tskIdx);
            % get the timestamp of when the task ended
            realEndTimeTsk = realTimeTable.LabelEndTimeRefArpStart(tskIdx);

            % get the index of the row with the timestamps of unexpected events
            % marked as important
            unexpectedIdx = contains(realTimeTable.Label,"_IMPORTANT");
            % get the timestamp of when the unexpected events started
            allUnexpectedEventStart = ...
                realTimeTable.LabelStartTimeRefArpStart(unexpectedIdx);
            % get the timestamp of when the unexpected events ended
            allUnexpectedEventEnd = ...
                realTimeTable.LabelEndTimeRefArpStart(unexpectedIdx);
            % check if there were any important unexpected events during the
            % baseline
            unexpectedEventBaselineStartTime = ...
                allUnexpectedEventStart(allUnexpectedEventStart > ...
                startTimeBaseline(end) & ...
                allUnexpectedEventStart < endTimeBaseline(1));
            unexpectedEventBaselineEndTime = ...
                allUnexpectedEventEnd(allUnexpectedEventStart > ...
                startTimeBaseline(end) & ...
                allUnexpectedEventStart < endTimeBaseline(1));
            % if there were no important unexpected events, keep the same
            % baseline timestamps
            if isempty(unexpectedEventBaselineStartTime) 
                newStartTimeBaseline = startTimeBaseline(end);
                newEndTimeBaseline = endTimeBaseline(1);
            else
                % otherwise split the baseline into before and after the
                % unexpected events, and take the bigger segment as the
                % baseline
               if (unexpectedEventBaselineStartTime(1) - startTimeBaseline(end)) > ...
                       (endTimeBaseline(1) - unexpectedEventBaselineEndTime(end))
                   newStartTimeBaseline = startTimeBaseline(end);
                   newEndTimeBaseline = unexpectedEventBaselineStartTime(1);
               else
                   newStartTimeBaseline = unexpectedEventBaselineStartTime(1);
                   newEndTimeBaseline = endTimeBaseline(1);
               end
            end
            % store the real durations of the baselines and tasks 
            realBaselineTime(tsk) = endTimeBaseline(1) - startTimeBaseline(end);
            realTskTime(tsk) = realEndTimeTsk - startTimeTsk;

            % take the 180-second uniform baseline segment from the middle of 
            % the real baseline recording (to account for recovery from the
            % task at the beginning of the baseline and anticipation at the end
            % of the baseline)
            uniMidTimeBaseline = newStartTimeBaseline + ...
                (newEndTimeBaseline - newStartTimeBaseline)/2;
            uniStartTimeBaseline = uniMidTimeBaseline - uniBaselineTime/2;
            uniEndTimeBaseline = uniMidTimeBaseline + uniBaselineTime/2;

            % check that the difference between the real task duration and the
            % intended task duration is not too large
            if abs(realTskTime(tsk) - intendedTskTimes(tsk)) < ...
                    threshDiffIntendedRealTskTime
                uniTskTime = intendedTskTimes(tsk);
            else
                % if the difference is too large, set the task duration used 
                % for the analysis to a number divisible by the
                % duration used for the individual segments in the analysis 
                uniTskTime = ...
                    floor(realTskTime(tsk)/divisibleTime)*divisibleTime;
            end

            % check that the difference between the end of the real task 
            % preparation time and the start of the task is not too large
            diffIntendedRealAnticipation(tsk) = endTimeTskPrep(1) - startTimeTsk;
            if abs(diffIntendedRealAnticipation(tsk)) < ...
                    threshDiffIntendedRealAnticipation
                % the start time of the anticipation used for the analysis is the 
                % fixed uniform anticipation duration substracted from the 
                % real task start time
                uniStartTimeAnticipation = startTimeTsk - uniAnticipationTime;
                % the end time of the anticipation used for the analysis is the
                % real task start time
                uniEndTimeAnticipation = startTimeTsk;
            else
                % if the difference is too large, set the anticipation end time
                % as the fixed uniform anticipation duration substracted from 
                % the end of the task preparation time 
                uniStartTimeAnticipation = endTimeTskPrep(1) - uniAnticipationTime;
                uniEndTimeAnticipation = endTimeTskPrep(1);
            end

            % check if the anticipation should be excluded for this participant
            % and task
            anticipationInvalidIdx = ...
                strcmp(paths.fileNameMatchTable.Audio_name(pIdx),...
                anticipationInvalidSubs);

            if max(anticipationInvalidIdx) > 0 
                if strcmp(anticipationInvalidTasks(anticipationInvalidIdx),...
                        tskNames(tsk))
                    uniStartTimeAnticipation = NaN;
                    uniEndTimeAnticipation = NaN;
                end
            end

            % the task end time used for the analysis is the determined 
            % uniform task duration added to the real task start time
            uniEndTimeTsk = startTimeTsk + uniTskTime;
            % the start time of the recovery used for the analysis is the 
            % end of the uniform task segment (so it might include a short
            % period where the task was still taking place)
            uniStartTimeRecovery = uniEndTimeTsk;
            % the end time of the recovery used for the analysis is the fixed 
            % uniform recovery time added to the start time of the recovery
            uniEndTimeRecovery = uniStartTimeRecovery + uniRecoveryTime;

            % put these timestamps for the current task into the new table for
            % the current participant
            curIdx = height(uniTimeTable);
            uniTimeTable(curIdx+1,1) = array2table(uniStartTimeBaseline);
            uniTimeTable(curIdx+2,1) = array2table(uniStartTimeAnticipation);
            uniTimeTable(curIdx+3,1) = array2table(startTimeTsk);
            uniTimeTable(curIdx+4,1) = array2table(uniStartTimeRecovery);

            uniTimeTable(curIdx+1,2) = array2table(uniEndTimeBaseline);
            uniTimeTable(curIdx+2,2) = array2table(uniEndTimeAnticipation);
            uniTimeTable(curIdx+3,2) = array2table(uniEndTimeTsk);
            uniTimeTable(curIdx+4,2) = array2table(uniEndTimeRecovery);
        end
        uniTimeTable(:,3) = cellstr(newLabelNames);
    end
    uniTimeTable.Properties.VariableNames = ....
            {'LabelStartTimeRefArpStart', 'LabelEndTimeRefArpStart','Label'};
        
    if ~strcmp(saveFile,"loadWithoutSave")
        writetable(uniTimeTable,uniTimeTableFilePath, ...
            'Delimiter','\t','WriteVariableNames', 0)
    end

    if strcmp(output,'uniformLength')
        varargout{1} = uniTimeTable;
    elseif strcmp(output,'originalLength')
        varargout{1} = uniTimeTable;
        varargout{2} = realTimeTable;
    else
        varargout{1} = uniTimeTable;
        varargout{2} = realTimeTable;
        varargout{3} = diffIntendedRealAnticipation;
        varargout{4} = realTskTime;
    end
    
end
%%
