#include "chess/shuffling_trainer.h"

#include "absl/time/time.h"
#include "chess/tensors.h"

namespace chess {

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

void ShufflingTrainer::Train(std::unique_ptr<TrainingSample> sample) {
  absl::MutexLock lock(&mu_);
#if 0
  if (b.ply() < 10) {
    // Discard half of early states to increase variance.
    if (rng_() % 2 == 0) {
      return;
    }
  }
#endif

  data_.push_back(std::move(sample));
  ++since_full_flush_;
  if (since_full_flush_ > 10 * shuffle_size_) {
    Flush();
    since_full_flush_ = 0;
  }
  while (data_.size() > max_size_) {
    // std::cerr << "Dropping from training queue\n";
    data_.pop_front();
  }
}

void ShufflingTrainer::Flush() {
  // XXX: Implement
}

void ShufflingTrainer::WorkerThread() {
  tensorflow::Tensor board_tensor = MakeBoardTensor(batch_size_);
  tensorflow::Tensor move_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size_, kMoveVectorSize}));
  tensorflow::Tensor value_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({batch_size_}));

  absl::MutexLock lock(&mu_);
  while (true) {
    const auto stopped_or_have_work = [this] {
      return stopped_ || data_.size() >= shuffle_size_;
    };
    mu_.Await(absl::Condition(&stopped_or_have_work));
    // TODO: This can be optimized to not hold the lock when building tensors.
    if (stopped_) {
      return;
    }
    for (int i = 0; i < batch_size_; ++i) {
      std::uniform_int_distribution<int> dist(0, data_.size() - 1);
      const int x = dist(rng_);
      auto board_matrix = board_tensor.SubSlice(i);
      BoardToTensor(data_[x]->board, &board_matrix);

      auto move_vec = move_tensor.SubSlice(i);
      for (int i = 0; i < kMoveVectorSize; ++i) {
        move_vec.vec<float>()(i) = 0.0;
      }
      const Color turn = data_[x]->board.turn();
      for (const auto& move : data_[x]->moves) {
        move_vec.vec<float>()(EncodeMove(turn, move.first)) = move.second;
      }

      value_tensor.flat<float>()(i) = data_[x]->value;

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

}  // namespace chess
