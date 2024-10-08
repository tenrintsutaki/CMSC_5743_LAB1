#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <ctime>   // For time

void im2col(const float* input, int height, int width, int channels,
            int kernel_size, int stride, int padding,
            std::vector<float>& output)
{
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    int channel_size = height * width;
    int output_col_size = channels * kernel_size * kernel_size;
    output.resize(out_height * out_width * output_col_size);

    int output_index = 0;
    int input_index = 0;

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
                        input_index = c * channel_size + h_in * width + w_in;

                        int current_channel_start = (c * kernel_size * kernel_size);
                        int row_start = (h * out_width + w) * output_col_size;
                        int kernel_start = (kh * kernel_size + kw);
                        output_index = current_channel_start + row_start + kernel_start;

                        if (h_in < height && w_in < width)
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

void convolution(const std::vector<float>& im2col_data,
                 const std::vector<float>& kernels,
                 int out_channels, int kernel_size,
                 int out_height, int out_width,
                 std::vector<float>& output)
{
    output.resize(out_channels * out_height * out_width, 0.0f);

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
                        int kernel_index = c * kernel_size * kernel_size + kh * kernel_size + kw;
                        int im2col_index = (h * out_width + w) * (kernel_size * kernel_size) + (kh * kernel_size + kw);
                        int output_index = c * out_height * out_width + h * out_width + w;
                        output[output_index] += im2col_data[im2col_index] * kernels[kernel_index];
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

     im2col(input.data(), height, width, channels, kernel_size, stride, padding, im2col_data);

    std::vector<float> output;

     convolution(im2col_data, kernel, out_channels, kernel_size,
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
    const int height = 7;
    const int width = 7;
    const int channels = 1;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;
    const int out_channels = 1;

    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Initialize input with random values
    std::vector<float> input(channels * height * width);
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
    im2col(input.data(), height, width, channels, kernel_size, stride, padding, im2col_data);

    // Perform convolution
    std::vector<float> output;
    std::cout << "im2col convert output size:" << im2col_data.size() << std::endl;
    convolution(im2col_data, kernels, out_channels, kernel_size,
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
