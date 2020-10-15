#include "generic/shuffling_trainer.h"

#include "absl/time/time.h"

namespace generic {

namespace {
constexpr bool kKeepPool = true;
}

ShufflingTrainer::ShufflingTrainer(generic::Model* model,
                                   const generic::Board& model_board,
                                   int batch_size, int shuffle_size)
    : model_(model),
      batch_size_(batch_size),
      shuffle_size_(shuffle_size),
      num_moves_(model_board.num_possible_moves()),
      worker_([this] { WorkerThread(); })
{
  CHECK_GE(shuffle_size, batch_size);
  model_board.GetTensorShape(batch_size, &board_shape_);
}

ShufflingTrainer::~ShufflingTrainer() {
  {
    absl::MutexLock lock(&mu_);
    stopped_ = true;
  }
  worker_.join();
}

void ShufflingTrainer::Train(std::unique_ptr<Board> b,
                             PredictionResult target) {
  BoardData new_board;
  new_board.board = std::move(b);
  new_board.target = std::move(target);
  absl::MutexLock lock(&mu_);

  data_.push_back(std::move(new_board));
  ++since_full_flush_;
  if (since_full_flush_ > 10 * shuffle_size_) {
    Flush();
    since_full_flush_ = 0;
  }
  while (data_.size() > max_size_) {
    data_.pop_front();
  }
}

void ShufflingTrainer::Flush() {
  // XXX: Implement
}

void ShufflingTrainer::WorkerThread() {
  tensorflow::Tensor board_tensor(tensorflow::DT_FLOAT, board_shape_);
  tensorflow::Tensor move_tensor(
      tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size_, num_moves_}));
  tensorflow::Tensor value_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({batch_size_}));

  absl::MutexLock lock(&mu_);
  while (true) {
    const auto stopped_or_have_work = [this] {
      return stopped_ || data_.size() >= shuffle_size_;
    };
    mu_.Await(absl::Condition(&stopped_or_have_work));
    if (stopped_) {
      return;
    }
    for (int i = 0; i < batch_size_; ++i) {
      std::uniform_int_distribution<int> dist(0, data_.size() - 1);
      const int x = dist(rng_);
      data_[x].board->ToTensor(&board_tensor, i);
      
      auto move_matrix = move_tensor.matrix<float>();
      for (int j = 0; j < num_moves_; ++j) {
        move_matrix(i,j) = 0.0;
      }
      for (const auto [move, p] : data_[x].target.policy) {
        move_matrix(i, move) = p;
      }
      value_tensor.flat<float>()(i) = data_[x].target.value;
      if (!kKeepPool) {
        std::swap(data_[x], data_.back());
        data_.pop_back();
      }
    }
    mu_.Unlock();

    model_->RunTrainStep(board_tensor, move_tensor, value_tensor);
    num_trained_.fetch_add(batch_size_, std::memory_order_relaxed);

    // XXX: This is very hacky.
    absl::SleepFor(absl::Milliseconds(50));

    mu_.Lock();
  }
}

}  // namespace generic
