function [y] = mylog(x)
    y = NaN(size(x));
    for i=1:length(x)
        if x(i) > 0
            y(i) = log(x(i));
        elseif x(i) == 0
            y(i) = 0;
        else
            y(i) = -log(abs(x(i)));
        end
    end
end