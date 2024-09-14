#include "portaudio.h" // portaudio library uses camel case, so  this project will use camel case
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

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

    const vector<float> &get_input_buffer()
    {
        Pa_ReadStream(stream, input_buffer.data(), frames_per_buffer);
        return input_buffer;
    }

    void set_output_buffer(vector<float> output_buffer) const
    {
        Pa_WriteStream(stream, output_buffer.data(), frames_per_buffer);
    }
};

int main()
{
    audio_data audio(2, SAMPLE_RATE, FRAMES_PER_BUFFER); // audio object with device number 1, 2 channels, sample rate 44100, and frames per buffer 512

    while (true)
    {
        vector<float> input_buffer = audio.get_input_buffer();
        for (int i = 0; i < input_buffer.size(); i += 2)
        {
            printf("left: %f, right: %f\n", input_buffer[i], input_buffer[i + 1]);
        }
        audio.set_output_buffer(input_buffer);
    }

    return 0;
}