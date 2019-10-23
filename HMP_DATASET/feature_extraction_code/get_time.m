function time = get_time(filename)

tens = filename(26) - 48;
ones = filename(27) - 48;

time = 10 * tens + ones;

end