#include "portaudio.h" // portaudio library uses camel case, so  this project will use camel case
#include "kiss_fft/kiss_fftr.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cmath>

using std::string;

using std::vector;

const int SAMPLE_RATE = 44100;     // sample rate, the number of samples per second
const int FRAMES_PER_BUFFER = 512; // frames per buffer, the number of samples per buffer

class audio_data
{
private:
    PaError err;      // PaError is a struct that contains information about the error, most portaudio functions return a PaError object
    PaStream *stream; // PaStream is a struct that contains information about the stream
    PaStreamParameters input_parameters;
    PaStreamParameters output_parameters;
    vector<float> input_buffer; // input buffer, a vector of floats that contains the input audio data
    int sample_rate;
    int frames_per_buffer;

    // checks if there is an error in the portaudio library initialization or other processes
    void check_error(PaError err) const
    {
        if (err != paNoError) // paNoError is a value of the PaError object that means there is no error
        {
            printf("PortAudio error: %s\n", Pa_GetErrorText(err));
            Pa_Terminate();
            exit(1); // exit the program with an error code
        }
    }

    void start_stream(int device_num, int channel_num)
    {
        // input parameter
        input_parameters.device = device_num;                                                     // device index to be used
        input_parameters.channelCount = channel_num;                                              // number of channels, 2 for stereo, 1 for mono
        input_parameters.sampleFormat = paFloat32;                                                // sample format, paFloat32 is a 32-bit floating point format
        input_parameters.suggestedLatency = Pa_GetDeviceInfo(device_num)->defaultLowInputLatency; // suggested latency for input
        input_parameters.hostApiSpecificStreamInfo = NULL;                                        // host API specific information, NULL for default

        // output parameter
        output_parameters.device = device_num;                                                     // device index to be used
        output_parameters.channelCount = channel_num;                                              // number of channels, 2 for stereo, 1 for mono
        output_parameters.sampleFormat = paFloat32;                                                // sample format, paFloat32 is a 32-bit floating point format
        output_parameters.suggestedLatency = Pa_GetDeviceInfo(device_num)->defaultLowInputLatency; // suggested latency for input
        output_parameters.hostApiSpecificStreamInfo = NULL;                                        // host API specific information, NULL for default

        // opening the stream with the input and output parameters
        // this builds the stream object with the input and output parameters
        err = Pa_OpenStream(&stream, &input_parameters, &output_parameters, sample_rate, frames_per_buffer, paNoFlag, NULL, NULL); // Pa_OpenStream() opens a stream with the input and output parameters
        check_error(err);

        // starting the stream
        err = Pa_StartStream(stream); // Pa_StartStream() starts the stream
        check_error(err);
    }

public:
    // constructors
    audio_data(int device_num, int channel_num, int sample_rate, int frames_per_buffer)
    {
        this->sample_rate = sample_rate;
        this->frames_per_buffer = frames_per_buffer;

        err = Pa_Initialize(); // Pa_Initialize() initializes the portaudio library, it returns a PaError object
        // if the initialization is successful, it returns paNoError value of the object
        // if the initialization is unsuccessful, it returns an error value
        check_error(err);

        start_stream(device_num, channel_num);

        input_buffer.resize(frames_per_buffer * channel_num); // resizing the input buffer to the size of the frames per buffer and the number of channels
    }

    audio_data(int channel_num, int sample_rate, int frames_per_buffer)
    {
        this->sample_rate = sample_rate;
        this->frames_per_buffer = frames_per_buffer;

        err = Pa_Initialize(); // Pa_Initialize() initializes the portaudio library, it returns a PaError object
        // if the initialization is successful, it returns paNoError value of the object
        // if the initialization is unsuccessful, it returns an error value
        check_error(err);

        print_device_info();

        int device_num;

        std::cout << "Enter device number: ";
        std::cin >> device_num;

        start_stream(device_num, channel_num);

        input_buffer.resize(frames_per_buffer * channel_num); // resizing the input buffer to the size of the frames per buffer and the number of channels
    }

    // destructor
    ~audio_data()
    {
        // stopping the stream
        err = Pa_StopStream(stream); // Pa_StopStream() stops the stream
        check_error(err);

        err = Pa_CloseStream(stream); // Pa_CloseStream() closes the stream
        check_error(err);

        err = Pa_Terminate(); // Pa_Terminate() terminates the portaudio library that was initialized by Pa_Initialize()
        check_error(err);
    }

    // prints the device information into the terminal
    void print_device_info() const
    {

        int num_devices = Pa_GetDeviceCount(); // Pa_GetDeviceCount() returns the number of devices found
        // it returns -1 if there is an error
        printf("Number of devices: %d\n", num_devices);
        printf(" \n");
        if (num_devices < 0)
        {
            printf("Error getting devices found\n");
            exit(1); // exit the program because it cant loop to get the device info
        }
        else if (num_devices == 0)
        {
            printf("No devices found\n");
            exit(0); // exit the program because it cant loop to get the device info
            // is a success code, but it just means that there are no devices found
        }

        const PaDeviceInfo *device_info; // a constant pointer because Pa_GetDeviceInfo() returns a constant pointer
        // PaDeviceInfo is a struct that contains information about the device
        for (int i = 0; i < num_devices; i++)
        {
            device_info = Pa_GetDeviceInfo(i);
            printf("Device %d: %s\n", i, device_info->name);
            printf("Max input channels: %d\n", device_info->maxInputChannels);
            printf("Max output channels: %d\n", device_info->maxOutputChannels);
            printf("Default sample rate: %f\n", device_info->defaultSampleRate);
            printf(" \n");
        }
    }

    // returns the input buffer, which is the waveform data of microphone input audio frames
    const vector<float> &get_input_buffer()
    {
        Pa_ReadStream(stream, input_buffer.data(), frames_per_buffer);
        return input_buffer;
    }

    // writes waveform data to the output buffer which plays sound
    void set_output_buffer(vector<float> output_buffer) const
    {
        Pa_WriteStream(stream, output_buffer.data(), frames_per_buffer);
    }
};

// fft_data class that contains the Fast Fourier Transform (FFT) algorithm
class fft_data
{
private:
    kiss_fftr_cfg config;   // contains the configuration of the FFT algorithm
    kiss_fft_scalar *input; // a malloc, input for the FFT algorithm
    double nfft;            // the number of samples of the input
    kiss_fft_cpx *output;   // a malloc, output from the input into the FFT algorithm, the output is a complex number, and contains nfft/2+1 elements

public:
    fft_data(double nfft)
    {
        this->nfft = nfft;
        config = kiss_fftr_alloc(nfft, 0, NULL, NULL);
        input = (kiss_fft_scalar *)malloc(sizeof(kiss_fft_scalar) * nfft);
        output = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * (nfft / 2 + 1));
    }

    ~fft_data()
    {
        free(input);
        free(output);
        free(config);
    }

    // sets the input for the FFT algorithm, and populates the output
    void set_input(vector<float> &input_buffer, int channel_num)
    {
        for (int i = 0; i < input_buffer.size(); i += channel_num)
        {
            input[i / channel_num] = input_buffer[i];
        }

        kiss_fftr(config, input, output);
    }

    // gets the raw output from the FFT algorithm
    kiss_fft_cpx *get_raw_output() const
    {
        return output;
    }

    // gets the amplitude (magnitude of the complex values) output from the FFT algorithm
    vector<float> get_amplitude_output() const
    {
        vector<float> amplitudes;
        for (int i = 0; i < nfft / 2 + 1; i++)
        {
            float amplitude = sqrt(output[i].r * output[i].r + output[i].i * output[i].i); // magnitude of the complex number
            amplitudes.push_back(amplitude);
        }
        return amplitudes;
    }
};

// normalises all the values in the array so that the maximum value is 1 and minimum is 0
void normalise_array(vector<float> &array)
{
    float max_value = 0.0f;

    for (int i = 0; i < array.size(); i++)
    {
        if (array[i] > max_value)
        {
            max_value = array[i];
        }
    }

    // divides all values by the maximum values to normalise the values
    // (meaning that largest value in the array will have a value of 1)
    if (max_value > 0)
    {
        for (int i = 0; i < array.size(); i++)
        {
            array[i] = array[i] / max_value;
        }
    }
}

// gets frequency values in an exponential scale
float exponential_freq_width(int max_freq, int min_freq, int num_bins, int x)
{
    // translate the graph with an x translation to make the first value (0) be the min_freq
    float x_translation = log10(min_freq) * num_bins / log10(max_freq);

    // scale the x value to the number of bins, y(x_increment * num_bins) = max_freq
    float x_increment = (num_bins - x_translation) / num_bins;
    x = x * x_increment;

    // scale the y value to the frequency range, each x increments increases the frequency by a power of 10, until it reaches the max frequency at num_bins
    float y = pow(10, (log10(max_freq) * (x + x_translation)) / num_bins);

    return y;
}

// converts the amplitude values to frequency bins
vector<float> to_freq_bins(vector<float> amplitudes, int num_bins, int sample_rate)
{
    int max_freq = sample_rate / 2;                          // the maximum frequency is half of the sample rate
    float amp_freq_increment = max_freq / amplitudes.size(); // the increment of the frequency for each amplitude
    // 0th index is 0, 1st index is 1 * amp_freq_increment, 2nd index is 2 * amp_freq_increment, etc.
    // however, we can also consider each amplitude as a frequency range, with its frequecy value being the middle of the range
    // therefore, the frequency range of each amplitude is also amp_freq_increment

    int half_freq_range = amp_freq_increment / 2; // because amp_freq_increment is the range of each amplitude, we divide it by 2 to get the middle of the range
    int min_freq = half_freq_range;               // the minimum frequency is also the same as half of the frequency range

    vector<float> freq_bins; // the frequency bins, each bin is a range of frequencies with its amplitude value

    for (int i = 0; i < num_bins; i++)
    {
        // finding the start and end frequency of the bin
        float start_freq = exponential_freq_width(max_freq, min_freq, num_bins, i);
        float end_freq = exponential_freq_width(max_freq, min_freq, num_bins, i + 1);

        // storing the amplitudes of a frequency bin
        vector<float> bin_amplitudes;

        for (int j = 1; j < amplitudes.size(); j++)
        {
            float amp_freq = j * amp_freq_increment;         // middle frequency range value of the amplitude
            float min_amp_freq = amp_freq - half_freq_range; // start frequency of the amplitude
            float max_amp_freq = amp_freq + half_freq_range; // end frequency of the amplitude

            // skip to the next frequency bin if the amplitude frequency is greater than the end frequency
            if (min_amp_freq >= end_freq)
            {
                break;
            }

            // if the amplitude frequency is within the frequency bin, add the amplitude to the bin
            if (max_amp_freq >= start_freq && min_amp_freq <= end_freq)
            {
                // checking if the ampitude frequency is out of bounds of the frequency bin
                float amp_freq_outbounds = 0.0f;

                if (min_amp_freq < start_freq)
                {
                    amp_freq_outbounds += start_freq - min_amp_freq;
                }

                if (max_amp_freq > end_freq)
                {
                    amp_freq_outbounds += max_amp_freq - end_freq;
                }

                // average the amplitude in proportion to its frequency range that is in the bin
                float amp_freq_inbounds = amp_freq_increment - amp_freq_outbounds;

                // add the average amplitude that is in the bin, inside the bin
                bin_amplitudes.push_back(amplitudes[j] * (amp_freq_inbounds / amp_freq_increment));
            }
        }

        // if there are no amplitudes in the bin, add a 0 amplitude
        if (bin_amplitudes.size() == 0)
        {
            freq_bins.push_back(0.0f);
            continue;
        }

        // average the amplitudes collected in the amplitudes of a bin
        float bin_avg_amplitude = 0.0f;

        for (int j = 0; j < bin_amplitudes.size(); j++)
        {
            bin_avg_amplitude += bin_amplitudes[j];
        }
        bin_avg_amplitude /= bin_amplitudes.size();

        // storing the average of each bin inside the frequency bins
        freq_bins.push_back(bin_avg_amplitude);
    }

    return freq_bins;
}

int main()
{
    // number of channels to record (2 is stereo, 1 is mono)
    int channel_num = 2;

    audio_data audio(channel_num, SAMPLE_RATE, FRAMES_PER_BUFFER); // audio object with device number 1, 2 channels, sample rate 44100, and frames per buffer 512
    fft_data fft(FRAMES_PER_BUFFER);                               // object to store the FFT algorithm with nfft 512 (or the frames per buffer)

    while (true)
    {
        // getting the input buffer from the audio object, contains audio frames of recorded input (microphone)
        vector<float> input_buffer = audio.get_input_buffer();

        // putting the input buffer into the FFT algorithm
        fft.set_input(input_buffer, channel_num);

        // getting the amplitude output from the FFT algorithm
        vector<float> amplitudes_raw = fft.get_amplitude_output();

        // converting the amplitude values to frequency bins with exponential scale
        vector<float> amplitudes = to_freq_bins(amplitudes_raw, 200, SAMPLE_RATE);
        normalise_array(amplitudes); // nomalising the amplitude values so that the maximum value is 1 and minimum is 0

        string output;
        std::cout << "\033[H";

        for (int i = 0; i < amplitudes.size(); i++)
        {
            float amplitude = amplitudes[i];

            if (amplitude < 0.125)
            {
                output += "▁";
            }
            else if (amplitude < 0.25)
            {
                output += "▂";
            }
            else if (amplitude < 0.375)
            {
                output += "▃";
            }
            else if (amplitude < 0.5)
            {
                output += "▄";
            }
            else if (amplitude < 0.625)
            {
                output += "▅";
            }
            else if (amplitude < 0.75)
            {
                output += "▆";
            }
            else if (amplitude < 0.875)
            {
                output += "▇";
            }
            else
            {
                output += "█";
            }
        }
        std::cout << output << std::flush;

        // sleep to reduce CPU usage
        Pa_Sleep(50);
    }

    // calling the destructors of the audio and fft objects
    audio.~audio_data();
    fft.~fft_data();

    return 0;
}