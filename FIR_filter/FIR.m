% Filter Specifications
N = 100; % Filter length (number of taps)
Fs = 10000;  % Sampling frequency
Fp = 0.23*pi*Fs/(2*pi); % Passband edge frequency
Fst = 0.2*pi*Fs/(2*pi); % Stopband edge frequency
Ap = 0.01; % Passband ripple (maximum allowed)
Ast = 80; % Stopband attenuation (minimum required)

% Design the filter using Parks-McClellan algorithm
b = firpm(N-1, [0, Fst/(Fs/2), Fp/(Fs/2), 1], [1, 1, 0, 0], [10^(-Ast/20), 10^(-Ap/20)]);

% Frequency response of the filter
freqz(b, 1, 1024, Fs);

% Plot the magnitude response
title('Magnitude Response of Low-pass FIR Filter');
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
grid on;

% Verify the stopband attenuation
[h, f] = freqz(b, 1, 1024, Fs);
stopband_attenuation = -20*log10(abs(h(f >= Fst)));
min_stopband_attenuation = min(stopband_attenuation);
disp(['Minimum Stopband Attenuation: ', num2str(min_stopband_attenuation), ' dB']);

% Verify the passband ripple
passband_response = 20*log10(abs(h(f <= Fp)));
max_passband_ripple = max(abs(passband_response));
disp(['Maximum Passband Ripple: ', num2str(max_passband_ripple), ' dB']);