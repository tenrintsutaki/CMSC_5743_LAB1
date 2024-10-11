#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

void im2col(const float* input, int batch_size, int height, int width, int channels,
            int kernel_size, int stride, int padding,
            std::vector<float>& output)
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

void convolution(const std::vector<float>& im2col_data, int batch_size,
                 const std::vector<float>& kernels,
                 int out_channels, int kernel_size,
                 int out_height, int out_width,
                 std::vector<float>& output){

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

void test(){
    const int height = 7;
    const int width = 7;
    const int channels = 1;
    const int kernel_size = 3;
    const int batch_size = 1;
    const int stride = 1;
    const int padding = 0;
    const int out_channels = 1;

    std::vector<float> input =
            {0,0,0,0,0,0,0,
            0,2,2,1,1,2,0,
            0,2,0,1,1,0,0,
            0,2,0,1,2,0,0,
            0,1,1,1,1,1,0,
            0,0,0,1,0,2,0,
            0,0,0,0,0,0,0};

    std::vector<float> kernel =
            {1,0,0,1,1,1,1,0,-1};

    std::vector<float> ground_truth =
            {4,6,3,5,4,
            2,6,2,4,4,
            1,5,3,4,4,
            2,4,3,3,4,
            0,2,2,4,3};

    std::vector<float> im2col_data;

     im2col(input.data(),batch_size, height, width, channels, kernel_size, stride, padding, im2col_data);

    std::vector<float> output;

     convolution(im2col_data, batch_size,kernel, out_channels, kernel_size,
                 (height + 2 * padding - kernel_size) / stride + 1,
                 (width + 2 * padding - kernel_size) / stride + 1,
                 output);

    // Output results for verification
    if (output ==  ground_truth){
        std::cout << "Correct" << std::endl;
    }else{
        std::cout << "Wrong" << std::endl;
    }
}

int main()
{
    const int height = 56;
    const int width = 56;
    const int channels = 3;
    const int out_channels = 64;
    const int kernel_size = 3;
    const int batch_size = 1;
    const int stride = 1;
    const int padding = 0;

    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Initialize input with random values
    std::vector<float> input(batch_size * channels * height * width);
    for (auto& val : input)
    {
        val = static_cast<float>(std::rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Initialize kernels with random values
    std::vector<float> kernels(out_channels * channels * kernel_size * kernel_size);
    for (auto& val : kernels)
    {
        val = static_cast<float>(std::rand()) / RAND_MAX; // Random float between 0 and 1
    }

    std::vector<float> im2col_data;
    im2col(input.data(), batch_size, height, width, channels, kernel_size, stride, padding, im2col_data);

    // Perform convolution
    std::vector<float> output;
    std::cout << "im2col convert output size:" << im2col_data.size() << std::endl;
    convolution(im2col_data, batch_size, kernels, out_channels, kernel_size,
                (height + 2 * padding - kernel_size) / stride + 1,
                (width + 2 * padding - kernel_size) / stride + 1,
                output);

    // Output results for verification
    std::cout << "conv output size: " << output.size() << std::endl;
    std::cout << "First few values in output:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    test();

    return 0;
}
