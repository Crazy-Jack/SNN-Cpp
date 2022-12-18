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




// struct FloatingPool {
//     int size, dim;
//     float prob;
//     torch::Tensor store = torch::rand({size, dim});
//     int warmup_done = 0;
//     int num_insert = 0;

//     void insert(uint64_t PC, uint64_t address) {
//         double norm_pc = normalize(PC);
//         double norm_addr = normalize(address);
//         num_insert++;
//     }

//     FloatingPool(int size, int dim, float prob) : size(size), dim(dim), prob(prob) {};
// };

// int print_my_pool() {
//     Pool pool(3, 4, 0.5);
//     Pool *pool_ptr = &pool;
    
//     cout << pool_ptr->store.sizes() << endl;
// }



struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(256, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 2));
    torch::manual_seed(0);
    B = torch::rand({40, 256}) * 10; 
    // torch::optim::Adam optim(
    //     this->parameters(), torch::optim::AdamOptions(1e-1).beta1(0.5));
  }
//   // noise for sensitive signals 
//   torch::Tensor B = torch::rand({2, 256}); // input is 2 dim

  torch::Tensor forward(torch::Tensor x) {

    x = torch::matmul(x, B);
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    x = torch::nn::functional::softmax(x, torch::nn::functional::SoftmaxFuncOptions(1));
    return x;
  }

  torch::Tensor get_tensor_nn(uint64_t PC, uint64_t memory_addr, vector<uint64_t> pc_history) {
    
    vector<float> pc_all_hash_f;

    // cout << ((float) (memory_addr % (PCMAP_SIZE2 * (uint64_t)1e+3))) / (PCMAP_SIZE2 * 1e+3) << endl;
    pc_all_hash_f.push_back(((float) (PC % PCMAP_SIZE)) / (float) PCMAP_SIZE);
    pc_all_hash_f.push_back(((float) (PC % PCMAP_SIZE2)) / (float) PCMAP_SIZE2);
    pc_all_hash_f.push_back(((float) (PC % PCMAP_SIZE3)) / (float) PCMAP_SIZE3);
    pc_all_hash_f.push_back(((float) (PC % PCMAP_SIZE4)) / (float) PCMAP_SIZE4);

    pc_all_hash_f.push_back(((float) (memory_addr % (PCMAP_SIZE * (uint64_t)1e+3))) / (float)(PCMAP_SIZE * 1e+3));
    pc_all_hash_f.push_back(((float) (memory_addr % (PCMAP_SIZE2 * (uint64_t)1e+3))) / (float)(PCMAP_SIZE2 * 1e+3));
    pc_all_hash_f.push_back(((float) (memory_addr % (PCMAP_SIZE3 * (uint64_t)1e+3))) / (float)(PCMAP_SIZE3 * 1e+3));
    pc_all_hash_f.push_back(((float) (memory_addr % (PCMAP_SIZE4 * (uint64_t)1e+3))) / (float)(PCMAP_SIZE4 * 1e+3));

    
    // cout << pc_history << endl;
    if (pc_history.size() < 8) {
        for (int i=0; i < 8; i++) {
           
            pc_all_hash_f.push_back(((float) 0));
            pc_all_hash_f.push_back(((float) 0));
            pc_all_hash_f.push_back(((float) 0));
            pc_all_hash_f.push_back(((float) 0));
          
        };
    } else {
        for (int i=0; i < pc_history.size(); i++) {
            uint64_t pc_history_i = pc_history[i];
            
            // cout << PC_i_hash << endl;
            
            pc_all_hash_f.push_back(((float) (pc_history_i % PCMAP_SIZE)) / (float)PCMAP_SIZE);
            pc_all_hash_f.push_back(((float) (pc_history_i % PCMAP_SIZE2)) / (float)PCMAP_SIZE2);
            pc_all_hash_f.push_back(((float) (pc_history_i % PCMAP_SIZE3)) / (float)PCMAP_SIZE3);
            pc_all_hash_f.push_back(((float) (pc_history_i % PCMAP_SIZE4)) / (float)PCMAP_SIZE4);
        };
    }
    
    // cout << pc_all_hash_f << endl;
    // cout << pc_all_hash << endl;
    vector<float> pc_all_hash_f_new;
    // cout << pc_all_hash_f.size() << endl;
    for (int i=0; i<pc_all_hash_f.size(); i++){
        float inset;
        
        if (pc_all_hash_f[i] < 0) {
            inset = (float) 0.0;
        } else {
            inset = pc_all_hash_f[i];
        }
        inset = min((float)1.0, inset);
        
        pc_all_hash_f_new.push_back(inset);
    }

   
    torch::Tensor out = torch::from_blob(pc_all_hash_f_new.data(), { static_cast<uint64_t>(pc_all_hash_f_new.size()) }).view({1, -1});
    return out;

  }

  bool get_prediction(uint64_t PC, uint64_t memory_addr, vector<uint64_t> pc_history){
        // torch no grad 
        torch::NoGradGuard no_grad;

        torch::Tensor input_;
        torch::Tensor pred;
        bool prediction;
        vector<uint64_t> pc_history_;
        pc_history_ = pc_history;
        input_ = get_tensor_nn(PC, memory_addr, pc_history);
        pred = forward(input_);
        // cout << pred << endl;
        prediction = pred[0][0].item<float>() > 0.5 ? 1 : 0;

    return prediction;
  };
  void train_templete(uint64_t PC, uint64_t memory_addr, vector<uint64_t> pc_history, bool label){
        torch::Tensor input_;
        torch::Tensor pred;
        bool prediction;
        vector<uint64_t> pc_history_;
        pc_history_ = pc_history;
        input_ = get_tensor_nn(PC, memory_addr, pc_history);
        pred = forward(input_);

        torch::Tensor target = torch::tensor({(int)label}).to(torch::kLong);
        torch::Tensor loss = torch::nll_loss(pred, target);
        cout << "loss: " << loss.item<double>() << endl;
        loss.backward();

  };
  
  void increase(uint64_t PC, uint64_t memory_addr, vector<uint64_t> pc_history){
    for (int i=0; i < 20; i++) {
        train_templete(PC, memory_addr, pc_history, false);
    }
  };
  void decrease(uint64_t PC, uint64_t memory_addr, vector<uint64_t> pc_history){
    train_templete(PC, memory_addr, pc_history, true);
  };


  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::Tensor B;
  


};


int print_my_torch() {
    Net net;
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