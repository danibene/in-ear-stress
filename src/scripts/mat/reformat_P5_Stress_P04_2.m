% had memory issues running this from python
[data,fs] = audioread("Z:\Shared\Documents\RD\RD2\_AudioRD\datasets\Biosignals\CritiasStress\synchedOriginal\P04_Berangere_28-02-2022_V1prompt\part2\IEM_L.wav");
p = 1;
q = 6;
resampFs = fs*(p/q);
resampData = resample(data, p, q);
audiowrite("Z:\Shared\Documents\RD\RD2\_AudioRD\datasets\Biosignals\CritiasStress\data_derivatives\DB8k\P04\P5_Stress-P04_2-Ieml-Sig-Raw.wav", resampData, resampFs);