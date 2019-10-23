function zcr = zero_crossing_rate(y)

zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);  
zero_indices = zci(y);
zcr = max( size(zero_indices) ) / 30;

end