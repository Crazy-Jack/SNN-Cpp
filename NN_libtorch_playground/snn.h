#ifndef SNN_H
#define SNN_H

using namespace std;
#include <torch/torch.h>
#include <random>
#include <inttypes.h>

#include <cstdint>
#include <limits>

template <typename T>
constexpr double normalize (T value, uint64_t max_num, uint64_t min_num) {
  return value < 0
    ? -static_cast<double>(value) / min_num
    :  static_cast<double>(value) / max_num
    ;
}

#include <cstdint>
#include <limits>
#define REPR_BITS 64
#define PCMAP_SIZE 2048
#define PCMAP_SIZE2 256
#define PCMAP_SIZE3 1024
#define PCMAP_SIZE4 512

template <typename T>
constexpr double normalize (T value) {
  return value < 0
    ? -static_cast<double>(value) / std::numeric_limits<T>::min()
    :  static_cast<double>(value) / std::numeric_limits<T>::max()
    ;
}


struct SNet : torch::nn::Module {
  SNet() {
    fc1 = register_module("fc1", torch::nn::Linear(256, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 1));
    torch::manual_seed(0);
    B = torch::rand({1, 256}) * 10; 
  
  }

  torch::Tensor forward(torch::Tensor x) {

    x = torch::matmul(x, B);
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    // x = torch::nn::functional::softmax(x, torch::nn::functional::SoftmaxFuncOptions(1));
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::Tensor B;
  


};


int print_my_torch() {
    SNet net;
    // Define Optimizer
    // torch::optim::Adam optimizer(net->parameters(),
    //                            torch::optim::AdamOptions(1e-3).beta1(0.5));
    // traverse parameters
    // for (const auto& pair : net.named_parameters()) {
    //     std::cout << pair.key() << ": " << pair.value() << std::endl;
    // }
    // forward
    std::cout << net.forward(torch::rand({2, 2})) << std::endl;
}


#endif