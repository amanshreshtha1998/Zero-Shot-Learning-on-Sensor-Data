function real_val = to_real(coded_val)

g = 9.81;
real_val = -1.5*g + (coded_val/63)*3*g;
size(real_val);
end