function feature = get_feature(data)

%data = load ('F:\year 2\hpg\project\HMP_Dataset\raw_data\Brush_teeth\Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt');
size(data);
accX = to_real(data(:, 1));
accY = to_real(data(:, 2));
accZ = to_real(data(:, 3));

[m, n] = size(accX);


%************Initialization of the statistical values of the windows******% 
windowsize = 3000;

acc = ones(1, m);
acc = sqrt ( accX.^2 + accY.^2 + accZ.^2 );

till = m + 1 - windowsize ;
nw = floor( (till-1) / ( windowsize/2 -1 ) )  +  1;


avgX=zeros(1,nw);
avgY=zeros(1,nw);
avgZ=zeros(1,nw);

avgACC=zeros(1,nw);
maxACC=-3*ones(1,nw);
minACC=-3*ones(1, nw);
stdACC=zeros(1,nw);

 maxX=100*ones(1,nw);
 maxY=100*ones(1,nw);
 maxZ=100*ones(1,nw); 
 
 minX=-100*ones(1,nw); 
 minY=-100*ones(1,nw); 
 minZ=-100*ones(1,nw);
 
 stdX=zeros(1,nw);
 stdY=zeros(1,nw);
 stdZ=zeros(1,nw); 

 
 XYcorr=zeros(1,nw);
 YZcorr=zeros(1,nw); 
 ZXcorr=zeros(1,nw); 

 zcrX = zeros(1, nw);
 zcrY = zeros(1, nw);
 zcrZ = zeros(1, nw);
 
 slopeX = zeros(1, nw);
 slopeY = zeros(1, nw);
 slopeZ = zeros(1, nw);
 
 
 energy=zeros(1,nw);


 i=1; j=1;
 
 
 while i <= till
     
     last = i+windowsize-1;
     x = accX(i:last);
     y = accY(i:last);
     z = accZ(i:last);
     
     corrmatrix = corrcoef( [x, y, z] );
     
     XYcorr(j) = corrmatrix(1, 2);
     YZcorr(j) = corrmatrix(2, 3);
     ZXcorr(j) = corrmatrix(3, 1);
     
     avgX(j)=mean(x);
     stdX(j)=std(x);
     maxX(j)=max(x);
     minX(j)=min(x);
     slopeX(j)=local_slope(x); 
     zcrX(j)=zero_crossing_rate(x);
     
     
     avgY(j)=mean(y);
     stdY(j)=std(y);
     maxY(j)=max(y);
     minY(j)=min(y);
     slopeY(j)=local_slope(y);
     zcrY(j)=zero_crossing_rate(y); 
     
     
     avgZ(j)=mean(z);
     stdZ(j)=std(z);
     maxZ(j)=max(z);
     minZ(j)=min(z);
     slopeZ(j)=local_slope(z);
     zcrZ(j)=zero_crossing_rate(z);
     
     
     avgACC(j)=mean(acc(i:last));
     stdACC(j)=std(acc(i:last));
     maxACC(j)=max(acc(i:last));
     minACC(j)=min(acc(i:last));
     
     energy(j)=sum(abs(fft(acc(i:last))))/26; 
     %Energy is defined as the normalized summation of absolute values of
     %Discrete Fourier Transform of a windowed signal sequence
     
     i=i+windowsize/2-1;
     j=j+1;
end
 
% row = each window
% col = each feature value of the window
feature = [maxX.',minX.',avgX.',stdX.', slopeX.', zcrX.',     maxY.',minY.',avgY.',stdY.', slopeY.', zcrY.',          maxZ.',minZ.',avgZ.',stdZ.', slopeZ.', zcrZ.',      maxACC.',minACC.',avgACC.',stdACC.',     XYcorr.',YZcorr.',ZXcorr.',    energy.'];

%csvwrite('features.csv',feature);

end