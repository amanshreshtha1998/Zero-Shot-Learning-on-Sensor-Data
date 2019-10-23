function [] = read_files(directory, save_file_name)


filePattern = fullfile(directory, '*.txt');
files = dir(filePattern);
feature = [];
for i=1:length(files)
    filename = files(i);
    fn = filename.name;
    time =get_time(fn);
    path = directory;
    path = strcat(path, '\');
    path = strcat(path, fn);
    data = load(path);
    feat = get_feature_without_window(data, time);
    feature = [feature; feat];
    %'F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt');


end


feature_names = ['maxX', 'minX', 'avgX', 'stdX', 'slopeX', 'zcrX',     'maxY','minY','avgY','stdY', 'slopeY', 'zcrY',          'maxZ', 'minZ', 'avgZ', 'stdZ', 'slopeZ', 'zcrZ',      'maxACC', 'minACC', 'avgACC', 'stdACC',     'XYcorr', 'YZcorr','ZXcorr',    'energy', 'time'];
csvwrite(save_file_name,feature);

end