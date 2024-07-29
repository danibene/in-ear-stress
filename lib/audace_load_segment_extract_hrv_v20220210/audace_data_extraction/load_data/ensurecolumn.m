function [output] = ensurecolumn(input)
    if size(input,1) >= size(input,2)
        output = input;
    else
        output = input.';
    end
end