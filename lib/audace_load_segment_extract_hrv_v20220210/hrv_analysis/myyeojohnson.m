function [y] = myyeojohnson(x)
    lmbda = 0;

    y = NaN(size(x));
    for i=1:length(x)
        if x(i) >= 0
            y(i) = log(x(i)+1);
        else
            y(i) = -((-x(i) + 1).^(2 - lmbda) - 1) ./ (2 - lmbda);
        end
    end
end