function [z] = myxnormrefy(x,y,method)
    z = NaN(size(x));
    for i=1:length(x)
        if x(i) + y(i) == 0
            z(i) = 0;
        else
            if strcmp(method,"percChange")
                z(i) = ((x(i) - y(i))./(y(i))).*100;
            else
                z(i) = (x(i))./(x(i) + y(i));
            end
        end
    end
end