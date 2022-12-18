#include <torch/torch.h>
#include <iostream>


int main() {
    torch::Tensor tensor = torch::rand({1, 3});
    std::cout << (tensor > 0) << std::endl;
}



