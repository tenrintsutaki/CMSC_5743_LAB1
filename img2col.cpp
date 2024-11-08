#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include <vector>
using namespace std;

void im2col(const float* input, int batch_size, int height, int width, int channels,
            int kernel_size, int stride, int padding,
            vector<float>& output)
{
    int out_height = (height + 2 * padding - kernel_size) / stride + 1; // out h
    int out_width = (width + 2 * padding - kernel_size) / stride + 1; // out w

    int channel_size = height * width; // points per channel
    int output_col_size = channels * kernel_size * kernel_size;
    output.resize(batch_size * out_height * out_width * output_col_size);

    int output_index = 0;
    int input_index = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < out_height; ++h)
            {
                for (int w = 0; w < out_width; ++w)
                {
                    for (int kh = 0; kh < kernel_size; ++kh)
                    {
                        for (int kw = 0; kw < kernel_size; ++kw)
                        {
                            int h_in = h * stride + kh; // h index traced from original 2D Feature Map
                            int w_in = w * stride + kw; // w index traced from original 2D Feature Map
                            int in_batch_start = b * channel_size * channels;
                            input_index = in_batch_start + c * channel_size + h_in * width + w_in; // Calculate input index

                            int out_batch_start = b * out_height * out_width * output_col_size; // Locate Current Batch
                            int out_channel_start = (c * kernel_size * kernel_size); // Locate Current Channel
                            int row_start = (h * out_width + w) * output_col_size; // Locate Current Row
                            int kernel_start = (kh * kernel_size + kw); // Locate index in this Row
                            output_index = out_batch_start + out_channel_start + row_start + kernel_start;

                            if (h_in < height && w_in < width) // Check if the padding area from input feature map
                            {
                                output[output_index] = input[input_index];
                            }
                            else
                            {
                                output[output_index] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void convolution(const vector<float>& im2col_data, int batch_size,
                 const vector<float>& kernels,
                 int out_channels, int kernel_size,
                 int out_height, int out_width,
                 vector<float>& output){

    output.resize(batch_size * out_channels * out_height * out_width, 0.0f);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int c = 0; c < out_channels; ++c)
        {
            for (int h = 0; h < out_height; ++h)
            {
                for (int w = 0; w < out_width; ++w)
                {
                    for (int kh = 0; kh < kernel_size; ++kh)
                    {
                        for (int kw = 0; kw < kernel_size; ++kw)
                        {
                            int kernel_index = c * kernel_size * kernel_size + kh * kernel_size + kw; // Locate the point in kernel
                            int im2col_index = b * out_height * out_width * (kernel_size * kernel_size) + (h * out_width + w) * (kernel_size * kernel_size) + (kh * kernel_size + kw); // Locate the point in the im2col data
                            int output_index = b * out_channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                            output[output_index] += im2col_data[im2col_index] * kernels[kernel_index];
                        }
                    }
                }
            }
        }
    }
}

void convolution_winograd(const vector<float>& im2col_data, int batch_size,
                          const vector<float>& kernels,
                          int out_channels, int kernel_size,
                          int out_height, int out_width,
                          vector<float>& output) {

    output.resize(batch_size * out_channels * out_height * out_width, 0.0f);

    // Winograd transformation for 2x2 blocks
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < out_channels; ++c) {
            for (int h = 0; h < out_height; h += 2) {
                for (int w = 0; w < out_width; w += 2) {
                    for (int k = 0; k < kernel_size * kernel_size; k += kernel_size) {

                    // Load data blocks
                    float D00 = im2col_data[b * out_height * out_width * 4 + (h * out_width + w) * 4];
                    float D10 = im2col_data[b * out_height * out_width * 4 + ((h + 1) * out_width + w) * 4];
                    float D20 = im2col_data[b * out_height * out_width * 4 + (h * out_width + (w + 1)) * 4];
                    float D30 = im2col_data[b * out_height * out_width * 4 + ((h + 1) * out_width + (w + 1)) * 4];

                    // Load kernel vectors
                    float k0 = kernels[c * kernel_size * kernel_size + k];
                    float k1 = kernels[c * kernel_size * kernel_size + k + 1];
                    float k2 = kernels[c * kernel_size * kernel_size + k + 2];

                    // Winograd transformation
                    float M0 = (D00 - D20) * k0;
                    float M1 = (D10 + D20) * (k0 + k1 + k2) / 2.0f;
                    float M2 = (D20 - D10) * (k0 - k1 + k2) / 2.0f;
                    float M3 = (D10 - D30) * k2;

                    // Calculate result
                    float r0 = M0 + M1 + M2;
                    float r1 = M1 - M2 - M3;

                    // Store result
                    output[b * out_channels * out_height * out_width + c * out_height * out_width + h * out_width + w] += r0;
                    output[b * out_channels * out_height * out_width + c * out_height * out_width + (h + 1) * out_width + w] += r1;
                    }
                }
            }
        }
    }
}

int main()
{
    //Input parameters
    const int height = 56;
    const int width = 56;
    const int channels = 3;
    const int out_channels = 64;
    const int kernel_size = 3;
    const int batch_size = 1;
    const int stride = 1;
    const int padding = 0;

    srand(time(0));     // Seed random number generator
    vector<float> input(batch_size * channels * height * width);     // Initialize input with random values
    for (auto& val : input)
    {
        val = rand();
    }

    vector<float> kernels(out_channels * channels * kernel_size * kernel_size); // Initialize kernels with random values
    for (auto& val : kernels)
    {
        val = rand();
    }
    vector<float> im2col_data;
    im2col(input.data(), batch_size, height, width, channels, kernel_size, stride, padding, im2col_data); // Perform im2col

    vector<float> output_normal;
    cout << "im2col convert output size:" << im2col_data.size() << endl;
    convolution(im2col_data, batch_size, kernels, out_channels, kernel_size,
                (height + 2 * padding - kernel_size) / stride + 1,
                (width + 2 * padding - kernel_size) / stride + 1,
                output_normal);     // Perform convolution

    cout << "Normal Conv Output size: " << output_normal.size() << endl;
    cout << "Process Finished !" << endl;

    vector<float> output_winograd;
    convolution_winograd(im2col_data, batch_size, kernels, out_channels, kernel_size,
                         (height + 2 * padding - kernel_size) / stride + 1,
                         (width + 2 * padding - kernel_size) / stride + 1,
                         output_winograd);

    cout << "Winograd Conv Output size: " << output_winograd.size() << endl;
    cout << "Process Finished !" << endl;

    if (output_normal == output_winograd){
        cout << "Answer Correct" << endl;
    }else{
        cout << "Answer Wrong" << endl;
    }

    int i = 0;
    while (i < 50){
        cout << output_normal[i] << "Normal" << endl;
        cout << output_winograd[i] << "Wino" << endl;
        i ++;
    }

}
