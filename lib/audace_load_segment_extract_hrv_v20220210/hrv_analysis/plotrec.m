x = ibi(1:end-1);
y = ibi(2:end);

yTime = ibiTime(2:end);
yOnlySucc = y;
yOnlySucc(ismember(yTime,timeMissing)) = NaN;

succDiffWithNan = yOnlySucc - x;
succDiff = succDiffWithNan(~isnan(succDiffWithNan));

allIdx = (ibi > 0);
decIdx = [false;(succDiffWithNan > 0)];
accIdx = [false;(succDiffWithNan < 0)];
conIdx = [false;(succDiffWithNan==0)];
%%
devRefBaseline = sqrt(sum((ibi-ibiBaseline).^2)/(length(ibi)-1));
devDecRefBaseline = sqrt(sum((ibi(decIdx)-ibiBaseline).^2)/(length(ibi(decIdx))-1));
devAccRefBaseline = sqrt(sum((ibi(accIdx)-ibiBaseline).^2)/(length(ibi(accIdx))-1));

%%
desiredFs = 4;
interplMethod = 'linear';
[interplIbi,interplIbiTime] = interplibi(ibi,ibiTime,desiredFs,interplMethod);
decIdxArea = logical(interp1(ibiTime,double(decIdx),interplIbiTime,'nearest'));
accIdxArea = logical(interp1(ibiTime,double(accIdx),interplIbiTime,'nearest'));
conIdxArea = logical(interp1(ibiTime,double(conIdx),interplIbiTime,'nearest'));

%%
N = length(interplIbiTime);
a = interplIbiTime(1);
b = interplIbiTime(end) + 1/desiredFs;
% Number of divisions
% If we divide the interval [a, b] into N pieces ,
% each piece would have width
width = ( b - a ) / N
%%
figure
plot(ibiTime,ibi)
hold on
%width = desiredFs;
% The command below says to look at all values of x,
% starting from a, increasing by the variable ’width ’
% each time , until the final value where x = b - width
posIdxArea = (interplIbi-ibiBaseline > 0);
negIdxArea = (interplIbi-ibiBaseline < 0);

posAccIdxArea = (posIdxArea & accIdxArea);
posDecIdxArea = (posIdxArea & decIdxArea);
posConIdxArea = (posIdxArea & conIdxArea);
negAccIdxArea = (negIdxArea & accIdxArea);
negDecIdxArea = (negIdxArea & decIdxArea);
negConIdxArea = (negIdxArea & conIdxArea);
allIdxArea = {posAccIdxArea,posDecIdxArea,posConIdxArea,...
    negAccIdxArea,negDecIdxArea,negConIdxArea};

colorMat = [[0.4940 0.1840 0.5560];...
    [0.4660 0.6740 0.1880];...
    [0.3010 0.7450 0.9330];...
    [0.6350 0.0780 0.1840];...
    [0.3847 0.2839 0.4834];
    [0.4398 0.9302 0.9429]];

approxInteg = NaN(1,length(allIdxArea));
for c=1:length(allIdxArea)
% For each of the x values , we draw a rectangle
% with the lower left corner at coordinate (x, 0) ,
% width the variable ’width ’ , and height the value f(x).
    relIbi = interplIbi(allIdxArea{c});
    relIbiTime = interplIbiTime(allIdxArea{c});
    for i=1:length(relIbiTime)
        if relIbi(i)-ibiBaseline > 0
            rectangle ('Position', [relIbiTime(i) ibiBaseline ...
                width abs(relIbi(i)-ibiBaseline)] , 'EdgeColor' , colorMat(c,:));
            hold on
        else
            rectangle ('Position', [relIbiTime(i) relIbi(i) width ...
                abs(relIbi(i)-ibiBaseline)] , 'EdgeColor' , colorMat(c,:));
            hold on
        end
    end
    l(c) = line(NaN,NaN,'LineWidth',2,'LineStyle','-','Color',colorMat(c,:));
    approxInteg(c) = sum(abs(relIbi-ibiBaseline).* width);
end

legend(l,["posAcc","posDec","posCon","negAcc","negDec","negCon"])

hold off % Done plotting



%%
approximated_integral = sum(abs(interplIbi-ibiBaseline).* width)
trapz(interplIbiTime,interplIbi)

%%
figure
scatter(ibiTime(accIdx),60./(ibi(accIdx)./1000))
hold on
scatter(ibiTime(decIdx),60./(ibi(decIdx)./1000))

legend(["Acc","Dec"])