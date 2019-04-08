#include "c4cc/shuffling_trainer.h"

#include "absl/time/time.h"

namespace c4cc {

namespace {
  constexpr bool kKeepPool = true;
}

ShufflingTrainer::ShufflingTrainer(Model* model, int batch_size,
                                   int shuffle_size)
    : model_(model),
      batch_size_(batch_size),
      shuffle_size_(shuffle_size),
      worker_([this] { WorkerThread(); }) {
  CHECK_GE(shuffle_size, batch_size);
}

ShufflingTrainer::~ShufflingTrainer() {
  {
    absl::MutexLock lock(&mu_);
    stopped_ = true;
  }
  worker_.join();
}

void ShufflingTrainer::Train(const Board& b, const Prediction& target) {
  absl::MutexLock lock(&mu_);
  if (b.ply() < 10) {
    // Discard half of early states to increase variance.
    if (rng_() % 2 == 0) {
      return;
    }
  }

  data_.push_back({b, target});
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
  tensorflow::Tensor board_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({batch_size_, 84}));
  tensorflow::Tensor move_tensor(tensorflow::DT_FLOAT,
                                 tensorflow::TensorShape({batch_size_, 7}));
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
      BoardToTensor(data_[x].b, &board_tensor, i);
      for (int m = 0; m < 7; ++m) {
        move_tensor.matrix<float>()(i, m) = data_[x].target.move_p[m];
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

    absl::SleepFor(absl::Milliseconds(50));

    mu_.Lock();
  }
}

}  // namespace c4cc
