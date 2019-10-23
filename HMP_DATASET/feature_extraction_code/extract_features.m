function [] = extract_features()
%F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data
%F:\year 3\zsl\HMP_DATASET\extracted_features\ten_to_thirteen

%class0
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_0.csv';
read_files(directory, save_file_name);


% class1
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Climb_stairs';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_1.csv';
read_files(directory, save_file_name);

% class2
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Comb_hair';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_2.csv';
read_files(directory, save_file_name);



% class3
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Descend_stairs';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_3.csv';
read_files(directory, save_file_name);


% class4
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Drink_glass';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_4.csv';
read_files(directory, save_file_name);


% class5
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Eat_meat';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_5.csv';
read_files(directory, save_file_name);



% class6
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Eat_soup';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_6.csv';
read_files(directory, save_file_name);



% class7
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Getup_bed';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_7.csv';
read_files(directory, save_file_name);



% class8
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Liedown_bed';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_8.csv';
read_files(directory, save_file_name);




% class9
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Pour_water';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\zero_to_nine\features_class_9.csv';
read_files(directory, save_file_name);




% class10
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Sitdown_chair';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\ten_to_thirteen\features_class_10.csv';
read_files(directory, save_file_name);



% class11
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Standup_chair';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\ten_to_thirteen\features_class_11.csv';
read_files(directory, save_file_name);




% class12
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Use_telephone';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\ten_to_thirteen\features_class_12.csv';
read_files(directory, save_file_name);



% class13
directory = 'F:\year 3\zsl\HMP_DATASET\HMP_Dataset\raw_data\Walk';
save_file_name = 'F:\year 3\zsl\HMP_DATASET\extracted_features\ten_to_thirteen\features_class_13.csv';
read_files(directory, save_file_name);


end