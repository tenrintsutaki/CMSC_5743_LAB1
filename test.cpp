#include <iostream>
#include <vector>

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

    // TODO: Your own implementation to assign values for output vector.

    std::vector<float> output;

    // Output results for verification
    if (output ==  ground_truth){
        std::cout << "Correct" << std::endl;
    }else{
        std::cout << "Wrong" << std::endl;
    }
}
