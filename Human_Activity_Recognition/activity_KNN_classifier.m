clear all
close all
clc

%% Define parameters
S = 16; % Number of subjects
activities = {'Standing', 'Walk', 'Run'}; % Types of activities
numActivities = length(activities); % number of activities
R = 3; % Number of repeated measurements per activity

basePath = 'C:...'; % Path to the folder containing CSV files
baseFilename = sprintf('Raw Data.csv');
FeatureTypes = {'mean', 'RMS', 'stdev', 'min', 'max', 'p2p', 'firstPeak_freq'};
numFeatureTypes = length(FeatureTypes);

%% Feature extraction
% Preallocate the Feature Matrix
numFeatures = 1 + 3 * numFeatureTypes + 1; % 1 subject ID + 3 acc components (x, y, z) x numFeatureTypes + 1 label = 23 rows
featureMatrix = zeros(S * R * numActivities, numFeatures); % (S*R*numActivities) x (numFeatures) = total number of signals
rowIdx = 1; % Index to fill the feature matrix

% Loop through each subject, activity, and measurement
for s = 1:S % subject

    for a = 1:numActivities % activity
        activity = activities{a}; % Name of the activity

        for r = 1:3 % repeated measurement

            fileFolder = sprintf('%s_%d_%d', activity, s, r); % Generate the file name
            filepath = fullfile(basePath, fileFolder, baseFilename); % Full path to the file

            % Import the file
            rawData = readtable(filepath, 'VariableNamingRule', 'preserve');

            % FEATURE EXTRACTION (in time)
            % TIME DOMAIN FEATURES
            % 1. Average
            avg_X = abs(mean(rawData.("Acceleration x (m/s^2)")));
            avg_Y = abs(mean(rawData.("Acceleration y (m/s^2)")));
            avg_Z = abs(mean(rawData.("Acceleration z (m/s^2)")));

            % 2. Root Mean Square (RMS)
            rms_X = rms(rawData.("Acceleration x (m/s^2)"));
            rms_Y = rms(rawData.("Acceleration y (m/s^2)"));
            rms_Z = rms(rawData.("Acceleration z (m/s^2)"));

            % 3. Standard Deviation
            std_X = std(rawData.("Acceleration x (m/s^2)"));
            std_Y = std(rawData.("Acceleration y (m/s^2)"));
            std_Z = std(rawData.("Acceleration z (m/s^2)"));

            % 4. Min
            min_X = min(rawData.("Acceleration x (m/s^2)"));
            min_Y = min(rawData.("Acceleration y (m/s^2)"));
            min_Z = min(rawData.("Acceleration z (m/s^2)"));

            % 5. Max
            max_X = max(rawData.("Acceleration x (m/s^2)"));
            max_Y = max(rawData.("Acceleration y (m/s^2)"));
            max_Z = max(rawData.("Acceleration z (m/s^2)"));

            % 6. Peak-to-Peak Amplitude (over the whole signal)
            p2p_X = max_X - min_X;
            p2p_Y = max_Y - min_Y;
            p2p_Z = max_Z - min_Z;

            % FREQUENCY DOMAIN FEATURES
            % Magnitude of the specturm via FFT
            Fs = 1 / mean(diff(rawData.("Time (s)"))); % Sampling frequency
            firstPeak_FreqDom = zeros(1, 3); % Store the first peak frequencies for x, y, z

            for i = 2:4 % Iterate over x, y, z columns (2nd to 4th)
                signal = rawData{:, i}; % Extract the signal for x, y, or z
                N = length(signal); % Length of the signal

                % Compute FFT
                FFT_signal = fft(signal); % FFT of the signal ---> complex values
                P2 = abs(FFT_signal / N); % Magnitude of the FFT. It is a two-sided spectrum, symmetric around the Nyquist frequncy
                P1 = P2(1:N/2+1); % Single-sided spectrum
                P1(2:end-1) = 2 * P1(2:end-1); % Adjust amplitude: double it for all components except the f=0Hz and f = Nyquist frequency

                % Frequency vector
                f = Fs * (0:(N/2)) / N; % the first frequency is the fundamental frequency (i.e., spectral resolution) = 1/T;
                                        % the last frequency is fs/2 (i.e., the Nyquist frequency).

                % 7. First peak in the FFT spectrum

                % Dynamic threshold for peak detection

                % Minimum distance between peaks in Hz
                minPeakDistance = 0.4;
                % The maximum of the peak height is computed by excluding
                % the first term which often it's very high and it can
                % affect the detection of the peaks in the following freq.
                minPeakHeight = 0.2 * max(P1(2:end));

                % Use findpeaks to detect peaks
                [peaks, locs] = findpeaks(P1, f, 'MinPeakHeight', minPeakHeight, ...
                                          'MinPeakDistance', minPeakDistance);

                % Filter out peaks below a frequency threshold (e.g., 0.5 Hz)
                validIdx = locs >= 0.5;
                peaks = peaks(validIdx);
                locs = locs(validIdx);

                % Extract the "base" step frequency (first peak)
                % The second condition has the intention to discriminate
                % the very small peaks in standing activity signals
                if ~isempty(locs) && any(peaks>=0.2)
                    firstPeak_FreqDom(i-1) = locs(1); % Store the first peak frequency
                else
                    firstPeak_FreqDom(i-1) = 0; % No peak detected ---> step freq = 0 Hz
                end
            end

            % Fill the feature matrix
            featureMatrix(rowIdx, :) = [s, avg_X, avg_Y, avg_Z, ...
                                        rms_X, rms_Y, rms_Z, ...
                                        std_X, std_Y, std_Z, ...
                                        min_X, min_Y, min_Z, max_X, max_Y, max_Z, ...
                                        p2p_X, p2p_Y, p2p_Z, ...
                                        firstPeak_FreqDom, a]; % Store activity label

            rowIdx = rowIdx + 1; % Move to the next row
        end
    end
end

%% 3D Scatter Plot with Custom Feature Selection (Improved Visibility)
% List the available features (corresponding to columns in your featureMatrix)
featureNames = {'mean_X', 'mean_Y', 'mean_Z', 'RMS_X', 'RMS_Y', 'RMS_Z', ...
                'stdev_X', 'stdev_Y', 'stdev_Z', 'min_X', 'min_Y', 'min_Z' ...
                'max_X', 'max_Y', 'max_Z', 'p2p_X', 'p2p_Y', 'p2p_Z', 'firstPeak_freq_X', 'firstPeak_freq_Y', 'firstPeak_freq_Z'};

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
numTest = [1 2 3 4 5 6 7 8 9 10 11 12 13 14]*9; % Test N subjects x 9 acc data per subject
k = [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29];
accuracy = zeros(length(numTest),length(k));

for i=1:length(numTest)
    for j=1:length(k)
        trainData = featureMatrix(1:end-numTest(i), 2:end-1); % All but last "numTest" rows, exclude label column
        trainLabels = featureMatrix(1:end-numTest(i), end); % Labels for training data
        testData = featureMatrix(end-numTest(i)+1:end, 2:end-1); % Last 3 rows for testing, exclude label column
        testLabels = featureMatrix(end-numTest(i)+1:end, end); % Labels for testing data

        % Define and train the KNN model
        %k = k(j); % You can experiment with different values of k
        KNN_model = fitcknn(trainData, trainLabels, 'NumNeighbors', k(j)); % model training
        % Predict labels for the test data
        predictedLabels = predict(KNN_model, testData);

        % Calculate and display accuracy
        accuracy(i,j) = sum(predictedLabels == testLabels) / numTest(i) * 100;
    end
end

% Plot accuracy in function of numTest

figure()
for i=1:length(numTest)
    for j=1:length(k)
        if (accuracy(i,j)<=50)
        plot3(numTest(i),k(j),accuracy(i,j),'r*','LineWidth',2);
        hold on;
        else if (accuracy(i,j)>=85)
        plot3(numTest(i),k(j),accuracy(i,j),'g*','LineWidth',2);
        hold on;
        else
        plot3(numTest(i),k(j),accuracy(i,j),'k*','LineWidth',1);
        hold on;
        end
        end
    end
end
grid on;
xlabel('Test data')
ylabel('K')
zlabel('Accuracy [%]');
title('Correlation between accuracy, K and number of test data');

% Plane XZ (no k)
figure()
for i = 1:length(numTest)
    for j = 1:length(k)
        if (accuracy(i,j) <= 50)
            plot(numTest(i), accuracy(i,j), 'r*', 'LineWidth', 2);
        elseif (accuracy(i,j) >= 85)
            plot(numTest(i), accuracy(i,j), 'g*', 'LineWidth', 2);
        else
            plot(numTest(i), accuracy(i,j), 'k*', 'LineWidth', 1);
        end
        hold on;
    end
end
grid on;
xlabel('Test data');
ylabel('Accuracy [%]');
title('Test data number effect');
hold off;

% Plane YZ (no numTest)
figure()
for i = 1:length(numTest)
    for j = 1:length(k)
        if (accuracy(i,j) <= 50)
            plot(k(j), accuracy(i,j), 'r*', 'LineWidth', 2);
        elseif (accuracy(i,j) >= 85)
            plot(k(j), accuracy(i,j), 'g*', 'LineWidth', 2);
        else
            plot(k(j), accuracy(i,j), 'k*', 'LineWidth', 1);
        end
        hold on;
    end
end
grid on;
xlabel('K');
ylabel('Accuracy [%]');
title('K neighbors effect');
hold off;
% % Display results
% fprintf('Testing Results:\n');
% for i = 1:numTest
%     fprintf('Test Observation %d: True Label = %s, Predicted Label = %s\n', ...
%             i, activities{testLabels(i)}, activities{predictedLabels(i)});
% end
