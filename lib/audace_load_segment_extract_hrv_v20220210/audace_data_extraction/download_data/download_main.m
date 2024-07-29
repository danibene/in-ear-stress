%% set paths

% see if a folder called "audace_data_extraction" is in the current directory
% if not keep going to the parent directory
% until the main directory containing all of the code and data is found
baseDirPath = pwd;
while ~isfolder(fullfile(baseDirPath,'audace_data_extraction'))
    [baseDirPath,~,~] = fileparts(baseDirPath);
end

% if you want the code to be in a different location than the data 
% change the paths in the loadpaths function 

% add the code in the entire directory
addpath(genpath(baseDirPath))

% load the filepaths for existing data and for the code
paths = loadpaths(baseDirPath,'saveFile','saveNewFile');
%% download metadata
struct2download(paths.downloadLinks.metaFiles,paths.metaDirPath,paths.tempDirPath);
%% download ECG IBI data
% original IBI 
struct2download(paths.downloadLinks.ecgFiles,paths.ecgDirPath,paths.tempDirPath);
% missing data recovered on 20211216
struct2download(paths.downloadLinks.ecgAddedIbiFiles,paths.ecgDirPath,paths.tempDirPath);
%% download raw ECG data
struct2download(paths.downloadLinks.ecgRawFiles,paths.ecgDirPath,paths.tempDirPath);
%% download ARP data
struct2download(paths.downloadLinks.arpFiles,paths.arpDirPath,paths.tempDirPath);