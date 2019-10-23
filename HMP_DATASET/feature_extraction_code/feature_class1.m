function [] = feature_class1()

feature1 = [];
% f1
data_f1 = [];
data1 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt');
data_f1 = [data_f1; data1];

data2 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-04-11-13-29-54-brush_teeth-f1.txt');
data_f1 = [data_f1; data2];

data3 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-30-08-35-11-brush_teeth-f1.txt');
data_f1 = [data_f1; data3];

data4 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-30-09-36-50-brush_teeth-f1.txt');
data_f1 = [data_f1; data4];

data5 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-30-21-10-57-brush_teeth-f1.txt');
data_f1 = [data_f1; data5];

data6 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-31-15-16-47-brush_teeth-f1.txt');
data_f1 = [data_f1; data6];

data7 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-06-02-10-42-22-brush_teeth-f1.txt');
data_f1 = [data_f1; data7];

data8 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-06-02-10-45-50-brush_teeth-f1.txt');
data_f1 = [data_f1; data8];

data9 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-06-06-10-45-27-brush_teeth-f1.txt');
data_f1 = [data_f1; data9];

data10 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-06-06-10-48-05-brush_teeth-f1.txt');
data_f1 = [data_f1; data10];

feature_f1 = get_feature(data_f1);
feature1 = [feature1; feature_f1];




% m1
data1 =  load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-30-10-34-16-brush_teeth-m1.txt');
feature_m1 = get_feature(data1);
feature1 = [feature1; feature_m1];


% m2
data1 = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-05-30-21-55-04-brush_teeth-m2.txt');
feature_m2 = get_feature(data1);
feature1 = [feature1; feature_m2];

feature_names = ['maxX', 'minX', 'avgX', 'stdX', 'slopeX', 'zcrX',     'maxY','minY','avgY','stdY', 'slopeY', 'zcrY',          'maxZ', 'minZ', 'avgZ', 'stdZ', 'slopeZ', 'zcrZ',      'maxACC', 'minACC', 'avgACC', 'stdACC',     'XYcorr', 'YZcorr','ZXcorr',    'energy'];
csvwrite('feature_class1.csv',feature1);

end