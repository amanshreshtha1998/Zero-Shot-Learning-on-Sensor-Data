function slope = local_slope(y_val)

[m,~] = size (y_val);
freq = 32;
x_val = 0:1/freq: (m-1 )/freq;
x_val = transpose(x_val);
size(x_val);
p = polyfit(x_val, y_val, 1);
slope = p(1);

end