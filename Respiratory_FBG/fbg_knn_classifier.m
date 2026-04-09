clc
clear all
close all

colors = [
    0.8500, 0.3250, 0.0980;  % Rosso tenue
    0.4660, 0.6740, 0.1880;  % Verde
    0.0000, 0.4470, 0.7410;  % Blu
    0.9290, 0.6940, 0.1250;  % Giallo oro
    0.4940, 0.1840, 0.5560;  % Viola
];

%% Time feature extraction

S = 13; % Number of subjects
activities = {'NB', 'DB', 'BH'}; % Types of activities
% NB = Normal Breathing, DB = Deep Breathing, BH = Breath Holding
numActivities = length(activities);

basePath = 'C:...'; % Path to the folder containing CSV files
baseFilename = sprintf('FBG1_main.txt');

load('breathing_signals.mat'); % Data from other subjects

%% my strain plots

k = 0.79;

for i = 1:3
    signal = cut_data{5, i};
    N = length(signal);
    t = linspace(0, N*0.1, N);

    figure(i)
    legendEntries = cell(1, 5);
    for j = 1:5
        lambda_B = signal(1,j);
        strain = (signal(:, j) - lambda_B)/(lambda_B*k)*10^6;

        plot(t, strain, 'color', colors(j, :), 'LineWidth', 1.3);
        hold on; grid on;
        legendEntries{j} = sprintf('FBG %d', j);
    end
    xlabel('Time [s]');
    ylabel('\mu strain');
    title(sprintf('Personal data - Activity: %s', activities{i}));
    legend(legendEntries, 'Location', 'best'); % Aggiunta legenda
end

%% Divide the signals half by half to have more samples

breath_data = cell(2*S,numActivities);

for s = 1:S
    for a = 1:numActivities
        for f = 1:5

            N = length(cut_data{s,a});
            breath_data{s,a}(:,f) = cut_data{s,a}(1:round(N/2),f);
            breath_data{s+S,a}(:,f) = cut_data{s,a}(round(N/2)+1:end,f);

        end
    end
end


%% KNN Classification

R = 2;
FeatureTypes = {'RMS', 'stdev', 'min', 'max', 'range', 'firstPeak_freq'};
numFeatureTypes = length(FeatureTypes);
numFeatures = 1 + 5 * numFeatureTypes + 1; % 1 subject ID + 5 FBG components x numFeatureTypes + 1 label = 32 columns
featureMatrix = zeros(S * R * numActivities, numFeatures); % (S*R*numActivities) x (numFeatures) = total number of signals
rowIdx = 1; % Index to fill the feature matrix

%initialize features
RMS = zeros(1,5);
STD = zeros(1,5);
MAX = zeros(1,5);
MIN = zeros(1,5);
range = zeros(1,5);
firstPeak_FreqDom = zeros(1,5);

for s = 1:2*S % subject

    for a = 1:numActivities % activity
        activity = activities{a}; % Name of the activity

        for f = 1:5 %FBGs

            % FEATURE EXTRACTION (in time)
            % TIME DOMAIN FEATURES

            RMS(f) = rms(breath_data{s,a}(:,f)) -mean(breath_data{s,a}(:,f));
            STD(f) = std(breath_data{s,a}(:,f));
            MIN(f) = min(breath_data{s,a}(:,f)) -mean(breath_data{s,a}(:,f));
            MAX(f) = max(breath_data{s,a}(:,f)) -mean(breath_data{s,a}(:,f));
            range(f) = MAX(f) - MIN(f);

            % FREQUENCY DOMAIN FEATURES
            % we first compute the Magnitude of the specturm via FFT
            Fs = 10; % Sampling frequency

            signal = breath_data{s,a}(:,f);% - mean(breath_data{s,a}(:,f));
            N = length(signal); % Length of the signal

            % Compute FFT
            FFT_signal = fft(signal); % FFT of the signal ---> complex values
            P2 = abs(FFT_signal / N); % Magnitude of the FFT. It is a two-sided spectrum, symmetric around the Nyquist frequncy
            P1 = P2(1:N/2+1); % Single-sided spectrum
            P1(2:end-1) = 2 * P1(2:end-1); % Adjust amplitude: double it for all components except the f=0Hz and f = Nyquist frequency

            % Frequency vector
            freq = Fs * (0:(N/2)) / N;

            % Minimum distance between peaks in Hz
            minPeakDistance = 0.02;
            minPeakHeight = 0.3*max(P1(2:end));

            % Use findpeaks to detect peaks
            [peaks, locs] = findpeaks(P1, freq, 'MinPeakHeight', minPeakHeight,'MinPeakDistance',minPeakDistance);
            % Filter out peaks below a frequency threshold (e.g., 0.5 Hz)
            validIdx = locs >= 0.1;
            peaks = peaks(validIdx);
            locs = locs(validIdx);

            % Extract the "base" step frequency (first peak)
            % The second condition has the intention to discriminate
            % the very small peaks in standing activity signals
            if ~isempty(locs) %&& any(peaks>=0.005)
                firstPeak_FreqDom(f) = locs(1); % Store the first peak frequency
            else
                firstPeak_FreqDom(f) = 0; % No peak detected ---> step freq = 0 Hz
            end
        end

        % Fill the feature matrix
        featureMatrix(rowIdx, :) = [s, RMS, STD, MIN, MAX, range, firstPeak_FreqDom, a]; % Store activity label

        rowIdx = rowIdx + 1; % Move to the next row

    end
end

%% 3D Scatter Plot with Custom Feature Selection (Improved Visibility)
% List the available features (corresponding to columns in your featureMatrix)
featureNames = { 'RMS_1', 'RMS_2', 'RMS_3','RMS_4','RMS_5','stdev_1', 'stdev_2', 'stdev_3','stdev_4','stdev_5', 'min_1', 'min_2', 'min_3' ...
                'min_4','min_5', 'max_1', 'max_2','max_3','max_4','max_5', 'range_1', 'range_2', 'range_3','range_4','range_5','freq_peak_1','freq_peak_2','freq_peak_3','freq_peak_4','freq_peak_5'};

% Prompt user to select three features
fprintf('Available Features:\n');
for i = 1:length(featureNames)
    fprintf('%d: %s\n', i, featureNames{i});
end

% Select features by their index
feat1 = input('Select the index of the 1st feature: ');
feat2 = input('Select the index of the 2nd feature: ');
feat3 = input('Select the index of the 3rd feature: ');

% Extract the selected features from the feature matrix
x = featureMatrix(:, feat1 +1);
y = featureMatrix(:, feat2 +1);
z = featureMatrix(:, feat3 +1);
labels = featureMatrix(:, end); % Activity labels (assumed last column)

% Define colors and markers for different activities
colors = ['r', 'g', 'b']; % Red, Green, Blue for activities
markers = ['o', 's', 'd']; % Circle, Square, Diamond markers
markerSize = 80; % Larger marker size
lineWidth = 1.5; % Thicker marker edges

% Create a new figure for the 3D scatter plot
figure;
hold on;

% Plot points for each activity
for i = 1:numActivities
    % Extract points for the current activity
    activityIdx = (labels == i);

    % 3D scatter plot for the current activity
    scatter3(x(activityIdx), y(activityIdx), z(activityIdx),   markerSize, colors(i), markers(i), 'filled', ...
             'MarkerEdgeColor', 'k', 'LineWidth', lineWidth); % Thicker edges
end

% Customize the plot
xlabel(featureNames{feat1}, 'Interpreter', 'none');
ylabel(featureNames{feat2}, 'Interpreter', 'none');
zlabel(featureNames{feat3}, 'Interpreter', 'none');
title('3D Scatter Plot of Selected Features');
legend(activities, 'Location', 'best');
grid on;
view(3); % Ensure 3D view is active
hold off;

%% k-nearest neighbor (KNN) Classifier Implementation and Testing
% Split data into training and testing sets
test_ratio = 0.3;
total = size(featureMatrix,1);
numTest = round(test_ratio * total);

% shuffle the data
randomOrder = randperm(total);
shuffled_featureMatrix = featureMatrix(randomOrder, :);

k = [1 2 3 4 5 6 7 8 9 10];
accuracy = zeros(1,length(k));

for j=1:length(k)
    trainData = shuffled_featureMatrix(1:end-numTest, 2:end-1); % All but last "numTest" rows, exclude label column
    trainLabels = shuffled_featureMatrix(1:end-numTest, end); % Labels for training data
    testData = shuffled_featureMatrix(end-numTest+1:end, 2:end-1); % Last 3 rows for testing, exclude label column
    testLabels = shuffled_featureMatrix(end-numTest+1:end, end); % Labels for testing data

    % Define and train the KNN model
    KNN_model = fitcknn(trainData, trainLabels, 'NumNeighbors', k(j)); % model training
    % Predict labels for the test data
    predictedLabels = predict(KNN_model, testData);

    % Calculate and display accuracy
    accuracy(j) = sum(predictedLabels == testLabels) / numTest * 100;
end

% Plot accuracy in function of numTest

figure();
for j = 1:length(k)
    if (accuracy(j) <= 50)
        plot(k(j), accuracy(j), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
    elseif (accuracy(j) >= 80)
        plot(k(j), accuracy(j), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
    else
        plot(k(j), accuracy(j), 'k*', 'MarkerSize', 10, 'LineWidth', 1);
    end
    hold on;

    text(k(j), accuracy(j) - 1.5, sprintf('%.1f', accuracy(j)), 'FontSize', 10, 'HorizontalAlignment', 'center');
end

xlim([0 11])
grid on;
xlabel('K');
ylabel('Accuracy [%]');
title('Accuracy with different K neighbors');

%% Windowing time signals: Pre-operation

data = cell(S,numActivities);
cut_data = cell(S,numActivities);

for s = 1:S % subject
    for a = 1:numActivities % activity
        activity = activities{a};

            fileFolder = sprintf('Subject_%d_%s', s, activity); % Generate the file name
            filepath = fullfile(basePath, fileFolder, baseFilename); % Full path to the file

            % Import the file
            rawData = readtable(filepath);
            rawData.Properties.VariableNames = {'FBG1','FBG2','FBG3','FBG4','FBG5'};
            data{s,a} = rawData;

            % Cutting signals
            N = height(data{s,a});
            t = linspace(0,N*0.1,N); %time
            figure()

            for i=1:5 %cycle the FBGs

                y = table2array(data{s,a}(:,i));

                subplot(5,1,i)
                plot(t,y,'color',colors(i,:),'LineWidth',1.5);
                xlabel('Time [s]');
                ylabel('Wavelength [nm]');
                sgtitle(sprintf('Subject %d - Activity: %s',s,activity));
            end
            [x_points, ~] = ginput(2); % Ottieni due punti (x-coordinates)

            % Trova gli indici dei punti selezionati
            startIdx = find(t >= x_points(1), 1);
            endIdx = find(t >= x_points(2), 1);

            % Taglia il segnale
            for i=1:5
                y = table2array(data{s,a}(:,i));
                cut_data{s,a}(:,i)= y(startIdx:endIdx);
            end
    end
end

%save('breathing_signals.mat','cut_data');
