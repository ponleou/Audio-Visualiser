#include <portaudio.h>
#include "kiss_fft/kiss_fftr.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <ncurses.h>
#include <sndfile.h>
#include <chrono>

using std::string;
using std::vector;

// for timing of refreshes
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

class audio_data
{
private:
    PaError err;      // PaError is a struct that contains information about the error, most portaudio functions return a PaError object
    PaStream *stream; // PaStream is a struct that contains information about the stream
    PaStreamParameters input_parameters;
    PaStreamParameters output_parameters;
    vector<float> input_buffer; // input buffer, a vector of floats that contains the input audio data
    int sample_rate;            // sample rate of the audio device
    int frames_per_buffer;      // number of audio frames to collect at a time

    // checks if there is an error in the portaudio library initialization or other processes
    // static because it is a utility function that is not dependent on the object, also another static method is also accessing it
    static void check_error(PaError err)
    {
        if (err != paNoError) // paNoError is a value of the PaError object that means there is no error
        {
            printf("PortAudio error: %s\n", Pa_GetErrorText(err));
            Pa_Terminate();
            exit(1); // exit the program with an error code
        }
    }

    // starts the stream with the input and output parameters
    void start_stream(int device_num, int channel_num)
    {
        // input parameter
        input_parameters.device = device_num;                                                     // device index to be used
        input_parameters.channelCount = channel_num;                                              // number of channels, 2 for stereo, 1 for mono
        input_parameters.sampleFormat = paFloat32;                                                // sample format, paFloat32 is a 32-bit floating point format
        input_parameters.suggestedLatency = Pa_GetDeviceInfo(device_num)->defaultLowInputLatency; // suggested latency for input
        input_parameters.hostApiSpecificStreamInfo = nullptr;                                     // host API specific information, nullptr for default

        // output parameter
        output_parameters.device = device_num;                                                     // device index to be used
        output_parameters.channelCount = channel_num;                                              // number of channels, 2 for stereo, 1 for mono
        output_parameters.sampleFormat = paFloat32;                                                // sample format, paFloat32 is a 32-bit floating point format
        output_parameters.suggestedLatency = Pa_GetDeviceInfo(device_num)->defaultLowInputLatency; // suggested latency for input
        output_parameters.hostApiSpecificStreamInfo = nullptr;                                     // host API specific information, nullptr for default

        // opening the stream with the input and output parameters
        // this builds the stream object with the input and output parameters
        err = Pa_OpenStream(&stream, &input_parameters, &output_parameters, sample_rate, frames_per_buffer, paNoFlag, nullptr, nullptr); // Pa_OpenStream() opens a stream with the input and output parameters
        check_error(err);

        // starting the stream
        err = Pa_StartStream(stream); // Pa_StartStream() starts the stream
        check_error(err);
    }

public:
    // constructors
    audio_data(int device_num, int channel_num, int frames_per_buffer)
    {
        err = Pa_Initialize(); // Pa_Initialize() initializes the portaudio library, it returns a PaError object
        // if the initialization is successful, it returns paNoError value of the object
        // if the initialization is unsuccessful, it returns an error value
        check_error(err);

        this->sample_rate = Pa_GetDeviceInfo(device_num)->defaultSampleRate; // sample rate of the device
        this->frames_per_buffer = frames_per_buffer;

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
    // a static method so that it could be called before the object is created
    static void print_device_info()
    {
        PaError err = Pa_Initialize();
        check_error(err);

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

        err = Pa_Terminate();
        check_error(err);
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

    int get_sample_rate() const
    {
        return sample_rate;
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
        config = kiss_fftr_alloc(nfft, 0, nullptr, nullptr);
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

// acts as a outline for the dynamic window data, takes care of the border and padding fo the main dynamic window
class dynamic_window_outline_data
{
protected:
    int outline_height;
    int outline_width;
    // the outline keeps track of the position (dynamic window will exist inside the outline window)
    int pos_x;
    int pos_y;
    // padding for the dynamic window
    int padding_x;
    int padding_y;
    WINDOW *outline_window; // the outline window
    bool border;            // if true, a border will be drawn around the outline window
    string border_text;     // the text that will be displayed on the border (like a name or heading)

    dynamic_window_outline_data(int height, int width, int pos_y, int pos_x, int padding_y, int padding_x, bool border, string border_text = "")
    {
        this->outline_height = height;
        this->outline_width = width;
        this->padding_x = padding_x;
        this->padding_y = padding_y;
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->outline_window = newwin(height, width, pos_y, pos_x);
        this->border_text = border_text;

        // making sure newwin is successful
        if (this->outline_window == nullptr)
        {
            exit(1);
        }

        this->border = border;
        if (this->border)
        {
            box(outline_window, 0, 0);
            this->padding_x++;
            this->padding_y++;

            // border text
            wattr_on(outline_window, A_BOLD, nullptr);
            mvwprintw(outline_window, 0, padding_x, this->border_text.c_str());
            wattr_off(outline_window, A_BOLD, nullptr);
        }
    }

    ~dynamic_window_outline_data()
    {
        delwin(outline_window);
    }

    // updates the outline window with the new height and width
    void update_window(int height, int width)
    {
        this->outline_height = height;
        this->outline_width = width;
        wclear(outline_window);

        wresize(outline_window, this->outline_height, this->outline_width);
        if (this->border)
        {
            box(outline_window, 0, 0);
        }

        wrefresh(outline_window);
    }

    // updates the outline window with the new height, width, position y, and position x
    void update_window(int height, int width, int pos_y, int pos_x)
    {
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->outline_height = height;
        this->outline_width = width;

        wclear(outline_window);

        mvwin(outline_window, pos_y, pos_x);
        wresize(outline_window, this->outline_height, this->outline_width);
        if (this->border)
        {
            box(outline_window, 0, 0);

            // border text
            wattr_on(outline_window, A_BOLD, nullptr);
            mvwprintw(outline_window, 0, padding_x, this->border_text.c_str());
            wattr_off(outline_window, A_BOLD, nullptr);
        }

        wrefresh(outline_window);
        // update_window(height, width);
    }
};

// dynamic window data that contains all the contents of the window
// acts as a dynamic window that is sized and positioned based on the screen size, and adjusts to the screen size
class dynamic_window_data : private dynamic_window_outline_data
{
private:
    // the maximum height and width of the screen, and also the margin of the screen, used to calculate the ratio of the window
    int max_width;
    int max_height;
    int margin_x;
    int margin_y;

    // the ratio of the height, width, position y, and position x of the window, entered by the user
    double height_ratio;
    double width_ratio;
    double pos_x_ratio;
    double pos_y_ratio;

public:
    // the current height and width of the window
    int height;
    int width;
    // the window and its name
    WINDOW *window;
    string name; // the name is used to identify the window inside the tui_data class's vector of dynamic_window_data

    dynamic_window_data(int max_height, int max_width, int margin_y, int margin_x, double height_ratio, double width_ratio, double pos_y_ratio, double pos_x_ratio, int padding_y, int padding_x, bool border, string name = "", bool show_name = false)
        : dynamic_window_outline_data(0, 0, 0, 0, padding_y, padding_x, border)
    {
        this->height_ratio = height_ratio;
        this->width_ratio = width_ratio;
        this->pos_x_ratio = pos_x_ratio;
        this->pos_y_ratio = pos_y_ratio;
        this->name = name;
        this->margin_x = margin_x;
        this->margin_y = margin_y;

        // if show name is true, the border text will be the name of the window
        if (show_name)
        {
            this->border_text = this->name;
        }

        // set 0,0,0,0 because the position and demensions will be calculated in the update_dynamic_window function
        window = newwin(1, 1, 0, 0);

        // making sure newwin is successful
        if (window == nullptr)
        {
            exit(1);
        }

        // passing the max height and width to the update_dynamic_window function to calculate the window size and position
        update_dynamic_window(max_height, max_width);
    }

    ~dynamic_window_data()
    {
        delwin(window);
    }

    // updates the window with the new height, width, position y, and position x
    void update_dynamic_window(int max_height, int max_width)
    {
        // sets the new max height and width
        this->max_width = max_width;
        this->max_height = max_height;

        // calculates the position and demensions of the window based on the ratio of the height, width, position y, and position x
        int pos_y = (int)floor((pos_y_ratio * (double)(max_height - (margin_y * 2))) + margin_y);
        int pos_x = (int)floor((pos_x_ratio * (double)(max_width - (margin_x * 2))) + margin_x);
        int new_height = (int)floor(height_ratio * (double)(max_height - (margin_y * 2)));
        int new_width = (int)floor(width_ratio * (double)(max_width - (margin_x * 2)));

        // updates the outline window with the new height, width, position y, and position x
        dynamic_window_outline_data::update_window(new_height, new_width, pos_y, pos_x);

        this->height = new_height - (padding_y * 2);
        this->width = new_width - (padding_x * 2);

        wclear(window);

        // sets the position of the window inside the outline window
        mvwin(window, pos_y + padding_y, pos_x + padding_x);
        wresize(window, this->height, this->width);

        wrefresh(window);
    }
};

// tui_data class that contains all the windows and the screen
class tui_data
{
private:
    int min_x, min_y, max_x, max_y; // keeps track of the minimum and maximum x and y coordinates of the screen
    // the margin of the screen
    int margin_x;
    int margin_y;
    // the vector of dynamic window data, contains all the dynamic window data on the screen
    vector<dynamic_window_data *> windows;

public:
    tui_data(int margin_y, int margin_x)
    {
        initscr();             // initializes the screen on the terminal
        curs_set(0);           // hides the cursor
        cbreak();              // enables ctrl+c to exit the program
        noecho();              // user input is not displayed on the screen
        nodelay(stdscr, true); // getch() does not delay the program's sequence (does not wait for user input)

        getbegyx(stdscr, min_y, min_x); // gets the minimum y and x coordinates of the screen
        getmaxyx(stdscr, max_y, max_x); // gets the maximum y and x coordinates of the screen

        this->margin_x = margin_x + min_x;
        this->margin_y = margin_y + min_y;
    }

    ~tui_data()
    {
        for (int i = 0; i < windows.size(); i++)
        {
            delete windows[i];
        }
        endwin();
    }

    // adds a dynamic window to the vector of dynamic window data
    void add_dynamic_window(double height_ratio, double width_ratio, double pos_y_ratio, double pos_x_ratio, int padding_y, int padding_x, bool border, string name = "", bool show_name = false)
    {
        dynamic_window_data *new_window = new dynamic_window_data(max_y, max_x, margin_y, margin_x, height_ratio, width_ratio, pos_y_ratio, pos_x_ratio, padding_y, padding_x, border, name, show_name);
        windows.push_back(new_window);
    }

    // gets the dynamic window using the index
    const dynamic_window_data *get_window(int index) const
    {
        return windows[index];
    }

    // gets the dynamic window using the name
    const dynamic_window_data *get_window(string name) const
    {
        // return null if the name is empty
        if (name == "")
        {
            return nullptr;
        }

        for (int i = 0; i < windows.size(); i++)
        {
            if (windows[i]->name == name)
            {
                return windows[i];
            }
        }

        // return null if the name is not found
        return nullptr;
    }

    // updates the screen and all the dynamic windows
    void update()
    {
        getbegyx(stdscr, min_y, min_x);
        getmaxyx(stdscr, max_y, max_x);
        this->margin_x = margin_x + min_x;
        this->margin_y = margin_y + min_y;

        int max_height = max_y;
        int max_width = max_x;

        clear();
        refresh();

        for (int i = 0; i < windows.size(); i++)
        {
            windows[i]->update_dynamic_window(max_height, max_width);
        }
    }
};

// class to read audio files
class audio_file_data
{
private:
    SF_INFO audio_info;   // stores the info of the audio file
    SNDFILE *audio_file;  // a pointer to the audio files
    float *output_frames; // audio frames from an audio file are outputted into this pointer (malloc)
    int frames_per_buffer;
    sf_count_t current_frame; // stores the position of audio frame where the audio file is currently at (the others have been read)
    string file_path;
    vector<float> output_vector; // the output vector that contains the audio data, can be accessed through its object

    // set ups before reading audio files
    void audio_file_setup()
    {
        // close the current audio file if it is opened
        if (audio_file != nullptr)
        {
            sf_close(audio_file);
        }

        // open the new audio file and start from the beginning (current frame = 0)
        audio_file = sf_open(file_path.c_str(), SFM_READ, &audio_info);
        current_frame = 0;
    }

public:
    audio_file_data(int frames_per_buffer)
    {
        this->frames_per_buffer = frames_per_buffer;
        current_frame = 0;
        audio_file = nullptr;
        output_frames = {};
        file_path = "";
    }

    ~audio_file_data()
    {
        if (audio_file != nullptr)
        {
            sf_close(audio_file);
        }
        if (output_frames != nullptr)
        {
            free(output_frames);
        }
    }

    // process to read the audio file (outputs into the output vector)
    void read_file(int resampling_rate)
    {
        // always clear the output vector before reading the file
        output_vector.clear();

        // making sure the audio file is opened and can be read in sections (if not, the audio file cannot be used)
        if (audio_file == nullptr || !audio_info.seekable)
        {
            return;
        }

        // if the output frames was allocated before, free it
        if (output_frames != nullptr)
        {
            free(output_frames);
        }

        // allocate the output frames based on the resampling rate and the frames per buffer
        // note that the resampling rate is the rate that the audio file will be resampled to
        output_frames = (float *)malloc(sizeof(float) * frames_per_buffer * (int)ceil((double)audio_info.samplerate / (double)resampling_rate) * audio_info.channels);

        // the number of audio frames that will be read from the audio file to create one output buffer
        int audio_frames_per_buffer = (int)ceil((double)frames_per_buffer * ((double)audio_info.samplerate / (double)resampling_rate));

        // seek to the current frame position in the audio file, and start reading from there
        sf_count_t seeked = sf_seek(audio_file, current_frame, SEEK_SET);

        if (seeked > -1)
        {
            // read the audio file into audio frames as float
            sf_readf_float(audio_file, output_frames, audio_frames_per_buffer);
            current_frame += audio_frames_per_buffer;

            // if the audio file's sampling rate is higher, the ratio will be more than 1
            // if the audio file's sampling rate is lower, the ratio will be less than 1
            double resampling_ratio = (double)audio_info.samplerate / resampling_rate;

            // processing for each audio frames
            for (int i = 0; i < frames_per_buffer; i++)
            {
                // the current audio output frame index in relation to the audio file's sampling rate
                double resampled_index_ratio = i * resampling_ratio;

                // the 2 frames between the current ratio index (start and end audio frames) that will be interpolated
                int start_audio_frame = (int)floor(resampled_index_ratio);
                int end_audio_frame = (int)ceil(resampled_index_ratio);

                // in case end_audio_frame is out of bounds
                if (end_audio_frame >= audio_frames_per_buffer)
                {
                    end_audio_frame = start_audio_frame;
                }

                // the ratio that the current index is in between the start and end audio frames
                double ratio_between_audio_frames = resampled_index_ratio - start_audio_frame;

                // processing for each channels in one audio frame
                for (int ch = 0; ch < audio_info.channels; ch++)
                {
                    // the 2 frames between the current ratio index for one channel
                    int start_channel_frame = start_audio_frame * audio_info.channels + ch;
                    int end_channel_frame = end_audio_frame * audio_info.channels + ch;

                    // linear interpolation between the 2 frames
                    float frame_gradient = (output_frames[end_channel_frame] - output_frames[start_channel_frame]);
                    // the y difference between the 2 frames (the x difference is always 1 because the frames are 1 unit apart)

                    // finding the new y value according to the ratio between the 2 frames
                    float interpolated_sample = (frame_gradient * ratio_between_audio_frames) + output_frames[start_channel_frame];

                    // store each frame in the output vector
                    output_vector.push_back(interpolated_sample);
                }
            }
        }
    }

    // call to input an audio file
    void input_file(string file_path)
    {
        // if the file path is different from the current file path, restart the process for the new file
        // this block sets up the process to read the audio file
        if (file_path != this->file_path)
        {
            this->file_path = file_path;
            audio_file_setup();
        }
    }

    vector<float> get_output_frames()
    {
        return output_vector;
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
            float min_amp_freq = amp_freq - half_freq_range; // start `frequency of the amplitude
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
                // also collecting the lenght of the amplitude that is out of bounds
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

// determining which bins are labeled
void set_labeled_bins(vector<int> &labeled_bins, int num_bins, int scr_size)
{
    // the first and last bins are always labeled
    labeled_bins.push_back(0);
    labeled_bins.push_back(num_bins - 1);

    num_bins -= 2; // the first and last bins are always labeled

    // there will be a labeled bin in every 20 horizontal lines
    int number_of_labeled_bins = (int)floor(scr_size / 20);

    int space_between_bins = (int)floor((double)num_bins / (double)number_of_labeled_bins);

    for (int i = 0; i < number_of_labeled_bins; i++)
    {
        labeled_bins.push_back(space_between_bins * i);
    }
}

// fixing the audio frames data to be usable for visualisation
// the only thing it is doing is reducing the number of audio frames in the array to the number of bins by finding the average of the audio frames
vector<float> to_audio_waves(vector<float> audio_frames, int num_bins)
{
    vector<float> audio_waves;

    int frames_per_bin = (int)floor((double)audio_frames.size() / (double)num_bins);

    if (frames_per_bin < 1)
    {
        frames_per_bin = 1;
    }

    for (int i = 0; i < num_bins; i++)
    {
        float sum = 0.0f;

        for (int j = 0; j < frames_per_bin; j++)
        {
            sum += audio_frames[(i * frames_per_bin) + j];
        }

        float avg = sum / (float)frames_per_bin;

        audio_waves.push_back(avg);
    }

    return audio_waves;
}

vector<float> extract_single_channel(vector<float> audio_frames, int channel_num)
{
    if (channel_num == 1)
    {
        return audio_frames;
    }

    vector<float> single_channel_frames;

    for (int i = 0; i < audio_frames.size(); i += channel_num)
    {
        single_channel_frames.push_back(audio_frames[i]);
    }

    return single_channel_frames;
}

int main()
{
    const int FRAMES_PER_BUFFER = 1024; // frames per buffer, the number of samples per buffer
    const int CHANNEL_NUM = 2;          // number of channels to record (2 is stereo, 1 is mono)

    int refresh_time = 50;        // time to sleep in milliseconds before each run of the loop
    bool frequency_visual = true; // toggle for frequency visualiser
    bool waveform_visual = false; // toggle for waveform visualiser

    bool inserting_file_path = false; // if the user is entering a file path
    string audio_file_path = "";      // path to the audio file

    bool audio_playing = false; // if the audio is playing

    int device_num; // the number of the device to use
    audio_data::print_device_info();
    std::cout << "Enter device number: ";
    std::cin >> device_num;

    audio_data audio(device_num, CHANNEL_NUM, FRAMES_PER_BUFFER); // Initialize audio data
    fft_data fft(FRAMES_PER_BUFFER);                              // Initialize FFT
    tui_data tui(1, 1);                                           // Initialize TUI with ncurses
    audio_file_data audio_file(FRAMES_PER_BUFFER);                // Initialize audio file data

    // creating the dynamic windows for the TUI
    tui.add_dynamic_window(0.2, 0.2, 0.0, 0.0, 0, 2, true, "Visualiser Type", true);
    tui.add_dynamic_window(0.2, 0.5, 0.0, 0.2, 0, 2, true, "Menu", true);
    tui.add_dynamic_window(0.8, 1.0, 0.2, 0.0, 1, 1, true, "Visualiser", true);
    tui.add_dynamic_window(0.2, 0.3, 0.0, 0.7, 0, 2, true, "Insert File", true);

    // accessing each dynamic window object with its index
    const dynamic_window_data *type_v = tui.get_window(0);
    const dynamic_window_data *menu = tui.get_window(1);
    const dynamic_window_data *visualizer = tui.get_window(2);
    const dynamic_window_data *insert_file = tui.get_window(3);

    // if the terminal supports colors, set the colors for the visualiser
    if (has_colors())
    {
        start_color();
        init_pair(1, COLOR_BLACK, COLOR_GREEN);
        init_pair(2, COLOR_GREEN, COLOR_BLACK);
    }

    auto start_time = high_resolution_clock::now();

    tui.update();

    while (true)
    {
        audio_file.input_file(audio_file_path);                      // input audio file for reading
        audio_file.read_file(audio.get_sample_rate());               // read the audio file
        vector<float> input_buffer = audio_file.get_output_frames(); // get the audio file's output frames
        audio_playing = false;                                       // audio is not playing

        // only playing the sound if the audio file is not empty
        if (input_buffer.size() != 0)
        {
            audio.set_output_buffer(input_buffer);
            audio_playing = true; // audio is playing
        }

        string insert_file_control_text = "[p]";
        int insert_file_text_length = insert_file_control_text.length();
        string insert_file_text = " Enter file path: ";

        while (inserting_file_path)
        {
            char ch = getch();

            if (ch == 10) // Enter key
            {
                inserting_file_path = false;
            }
            else if (ch == 27) // Escape key
            {
                inserting_file_path = false;
                audio_file_path = "";
            }
            else if (ch == 127) // Backspace key
            {
                wclear(insert_file->window);

                if (audio_file_path.size() > 0)
                    audio_file_path.resize(audio_file_path.size() - 1);
            }
            else if (ch != ERR) // Backspace key
            {
                audio_file_path += static_cast<char>(ch);
            }

            if (has_colors())
                wattr_on(insert_file->window, COLOR_PAIR(2), nullptr);

            mvwprintw(insert_file->window, (insert_file->height - 1) / 3, 0, insert_file_control_text.c_str());

            wattr_off(insert_file->window, COLOR_PAIR(2), nullptr);

            mvwprintw(insert_file->window, (insert_file->height - 1) / 3, insert_file_text_length, insert_file_text.c_str());

            if (has_colors())
                wattr_on(insert_file->window, COLOR_PAIR(1), nullptr);
            else
                wattr_on(insert_file->window, A_STANDOUT, nullptr);

            mvwprintw(insert_file->window, insert_file->height / 3 * 2, 0, audio_file_path.c_str());

            wattr_off(insert_file->window, A_STANDOUT, nullptr);
            wattr_off(insert_file->window, COLOR_PAIR(1), nullptr);

            wrefresh(insert_file->window);
        }

        // delay refresh time to decrease CPU usage, only works if audio is playing
        // when audio is not playing, the section below has a sleep function to decrease CPU usage better, that section only works when audio is not playing
        if (duration_cast<std::chrono::milliseconds>(high_resolution_clock::now() - start_time).count() > refresh_time || !audio_playing)
        {
            if (audio_file.get_output_frames().size() == 0)
                input_buffer = audio.get_input_buffer(); // Get input buffer from microphone if audio file is empty

            tui.update(); // screen is cleared
            start_time = high_resolution_clock::now();

            // text for the menu section
            string quit_control_text = "[q]";
            string quit_text = " Quit";

            string switch_control_text = "[TAB]";
            string switch_text = " Switch Visuals";

            string refresh_control_text = "[+/-]";
            string refresh_text = " Refresh Time: " + std::to_string(refresh_time) + " ms";

            int switch_text_length = switch_text.length() + switch_control_text.length();
            int quit_text_length = quit_control_text.length() + quit_text.length();
            int refresh_text_length = refresh_control_text.length() + refresh_text.length();

            // printing the control text (it has colors if the terminal supports colors)
            if (has_colors())
                wattr_on(menu->window, COLOR_PAIR(2), nullptr);

            mvwprintw(menu->window, (menu->height - 1) / 2, 0, switch_control_text.c_str());
            mvwprintw(menu->window, (menu->height - 1) / 2, switch_text_length + 3, quit_control_text.c_str());
            mvwprintw(menu->window, (menu->height - 1) / 2, switch_text_length + 3 + quit_text_length + 3, refresh_control_text.c_str());

            wattr_off(menu->window, COLOR_PAIR(2), nullptr);

            // printing the menu text (without colors)
            mvwprintw(menu->window, (menu->height - 1) / 2, switch_control_text.length(), switch_text.c_str());
            mvwprintw(menu->window, (menu->height - 1) / 2, switch_text_length + 3 + quit_control_text.length(), quit_text.c_str());
            mvwprintw(menu->window, (menu->height - 1) / 2, switch_text_length + 3 + quit_text_length + 3 + refresh_control_text.length(), refresh_text.c_str());

            // text for the insert file section
            if (has_colors())
                wattr_on(insert_file->window, COLOR_PAIR(2), nullptr);

            mvwprintw(insert_file->window, (insert_file->height - 1) / 3, 0, insert_file_control_text.c_str());

            wattr_off(insert_file->window, COLOR_PAIR(2), nullptr);

            mvwprintw(insert_file->window, (insert_file->height - 1) / 3, insert_file_text_length, insert_file_text.c_str());
            mvwprintw(insert_file->window, insert_file->height / 3 * 2, 0, audio_file_path.c_str());

            // text for the visualiser type section
            // the current visualiser type is highlighted (with colours if supported)
            if (frequency_visual)
            {
                if (has_colors())
                    wattr_on(type_v->window, COLOR_PAIR(1), nullptr);
                else
                    wattr_on(type_v->window, A_STANDOUT, nullptr);
            }

            mvwprintw(type_v->window, (type_v->height - 1) / 3, 0, "Frequency Visualiser");

            wattr_off(type_v->window, A_STANDOUT, nullptr);
            wattr_off(type_v->window, COLOR_PAIR(1), nullptr);

            if (waveform_visual)
            {
                if (has_colors())
                    wattr_on(type_v->window, COLOR_PAIR(1), nullptr);
                else
                    wattr_on(type_v->window, A_STANDOUT, nullptr);
            }

            mvwprintw(type_v->window, type_v->height / 3 * 2, 0, "Waveform Visualiser");

            wattr_off(type_v->window, A_STANDOUT, nullptr);
            wattr_off(type_v->window, COLOR_PAIR(1), nullptr);

            int num_bins = visualizer->width; // Number of bins for the visualizer

            if (waveform_visual)
            {
                input_buffer = extract_single_channel(input_buffer, CHANNEL_NUM);
                // converting the audio frames from the input buffer to work as audio waves for the visualiser
                vector<float> audio_waves = to_audio_waves(input_buffer, num_bins);

                normalise_array(audio_waves); // Normalize the audio waves

                for (int i = 0; i < visualizer->width; i++)
                {
                    // when the audio wave's value is 0, it will be drawn in the middle of the window
                    int mid_height = (int)floor((double)(visualizer->height - 1) / 2.0);

                    // drawing the audio waves as displacement from the center/middle of the window
                    if (audio_waves.size() > i)
                    {
                        int height = (int)floor((double)mid_height * audio_waves[i]);
                        int next_height = 0;
                        if (i + 1 < audio_waves.size())
                        {
                            next_height = (int)floor((double)mid_height * audio_waves[i + 1]);
                        }
                        else
                        {
                            next_height = -(int)floor((double)mid_height * audio_waves[i - 1]);
                        }

                        // if the current height's gradient is positive, it will be drawn with a upward slope (/)
                        if (next_height > height)
                        {
                            mvwprintw(visualizer->window, mid_height - height, i, "/");
                        }

                        // if the current height's gradient is negative, it will be drawn with a downward slope (\)
                        if (next_height < height)
                        {
                            mvwprintw(visualizer->window, mid_height - height, i, "\\");
                        }

                        // if the current height's gradient is zero, it will be drawn with a horizontal line (-)
                        if (next_height == height)
                        {
                            mvwprintw(visualizer->window, mid_height - height, i, "-");
                        }

                        // drawing the line of 0 displacement
                        mvwprintw(visualizer->window, mid_height, i, "=");
                    }
                }
            }

            if (frequency_visual)
            {
                // Set input for FFT and get amplitude output
                fft.set_input(input_buffer, CHANNEL_NUM);
                vector<float> amplitudes_raw = fft.get_amplitude_output();

                // Convert amplitude values to frequency bins with exponential scale
                vector<float> frequency_bins = to_freq_bins(amplitudes_raw, num_bins, audio.get_sample_rate()); // Adjust the number of bins as needed
                normalise_array(frequency_bins);                                                                // Normalize amplitudes

                // getting the specific bins that will be labeled
                vector<int> label_bins;
                set_labeled_bins(label_bins, frequency_bins.size(), num_bins);

                // Draw visualizer bars
                for (int i = 0; i < frequency_bins.size(); i++)
                {
                    mvwprintw(visualizer->window, visualizer->height - 2, i, "=");
                    // Print line under the bars, on the second last time from the margin (window_margin + 2), leaving a space at the bottom

                    int height = (int)(frequency_bins[i] * (float)(visualizer->height - 2));

                    // drawing from top to bottom
                    for (int j = 0; j < height; j++)
                    {
                        // max_height is the bottom coordinate of the window, max_height - height is the top coordinate of the window for the bars
                        mvwprintw(visualizer->window, visualizer->height - 3 - j, i, "#"); // Print visualizer bars at position (row, column)
                    }
                }

                // section for the frequency labels
                for (int i = 0; i < label_bins.size(); i++)
                {
                    // getting the label for the frequency for the bin
                    int freq = (int)exponential_freq_width(audio.get_sample_rate() / 2, audio.get_sample_rate() / (2 * FRAMES_PER_BUFFER), num_bins, label_bins[i]);

                    int pos_x = label_bins[i];
                    int pos_y = visualizer->height - 1;

                    int max_text_width = 3;

                    if (pos_x + max_text_width > visualizer->width)
                    {
                        pos_x = visualizer->width - max_text_width;
                    }

                    if (freq < 1000)
                        mvwprintw(visualizer->window, pos_y, pos_x, "%d", freq);
                    else
                        mvwprintw(visualizer->window, pos_y, pos_x, "%dk", (freq / 1000));
                }
            }

            // Refresh the windows
            wrefresh(visualizer->window);
            wrefresh(type_v->window);
            wrefresh(menu->window);
            wrefresh(insert_file->window);
        }

        if (!audio_playing)
        {
            timeout(refresh_time); // if audio is not playing, we can wait for user input, functions as sleep to decrease CPU usage
        }
        else
        {
            nodelay(stdscr, true); // if audio is playing, we cannot wait for user input, so we need to keep the program running
        }

        // user input
        char key = getch();

        // toggle the visualiser type
        if (key == 9) // 9 is TAB in characters
        {
            frequency_visual = !frequency_visual;
            waveform_visual = !waveform_visual;
        }

        // change the sleep time
        if (key == '+')
        {
            refresh_time++;
        }
        if (key == '-')
        {
            refresh_time--;
        }

        if (key == 'p')
        {
            inserting_file_path = true;
        }

        // make sure the sleep time is not negative or too high (set at 150 max)
        if (refresh_time < 1)
        {
            refresh_time = 1;
        }

        // Check if a key is pressed to exit the loop
        if (key == 'q')
            break;
    }

    return 0;
}