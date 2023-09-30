function struct2download(files,dataDirPath,tempDirPath)
% Download the links in a struct, extracting zip files if the link text
% (should be the file name) ends in .zip
%
% IN	files                   struct with fields for the link (url) and
%                               link text (name)
%
%       dataDirPath             char containing path to the directory where
%                               the data will be saved
%
%       tempDirPath             char containing path to the directory where
%                               any temporary data (i.e. zip files) can be
%                               saved
%
% OUT	files                   struct with fields for the link (url) and
%                               link text (name)
%
% v20211213 DB

    % if the directories for the data and temporary files don't exist, 
    % create them
    if ~exist(dataDirPath,'dir')
        mkdir(dataDirPath)
    end
    if ~exist(tempDirPath,'dir')
        newTempDirPath = 1;
        mkdir(tempDirPath)
    else
        newTempDirPath = 0;
    end
    
    for i=1:length(files.name)
        [~,name,ext] = fileparts(files.name(i));
        % check whether the current file is a zip
        % by seeing if if the link text(should be the file name) 
        % ends in .zip
        if strcmp(ext,'.zip')
            % if so, download to a temporary directory 
            zipDirPath = fullfile(tempDirPath,name);
            targetDirPath = fullfile(dataDirPath,name);
            % check that the directory where the zip file will be extracted
            % doesn't already exist (a directory named the name of the 
            % link text without the extension, inside the data directory)
            if ~exist(targetDirPath,'dir')
                if ~exist(zipDirPath,'dir')
                    mkdir(zipDirPath)
                end
                disp(strcat('Downloading',{' '},files.name(i)))
                zipFilePath = fullfile(zipDirPath,files.name(i));
                % download the zip file from the URL
                websave(zipFilePath,files.url(i));
                disp(strcat('Unzipping',{' '},files.name(i)))
                % unzip the file to the data directory
                unzip(zipFilePath,dataDirPath);
                disp(strcat('Deleting temporary directory for',{' '},files.name(i)))
                % delete the zip file
                delete(zipFilePath)
                % remove the zip directory
                rmdir(zipDirPath)
            end      
        else
            targetFilePath = fullfile(dataDirPath,files.name(i));
            % if it is not a zip file, check that a file of the same name
            % is not already in the data directory
            if ~exist(targetFilePath,'file')
                disp(strcat('Downloading',{' '},files.name(i)))
                % download the file from the URL
                websave(targetFilePath,files.url(i));
            end
        end
    end
    % delete the temporary directory
    if newTempDirPath
        rmdir(tempDirPath)
    end
end
