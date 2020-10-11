#include "chess/shuffling_trainer.h"

#include "absl/time/time.h"
#include "chess/tensors.h"

namespace chess {

namespace {
constexpr bool kKeepPool = false;
}

ShufflingTrainer::ShufflingTrainer(generic::Model* model, int batch_size,
                                   int shuffle_size)
    : model_(model), batch_size_(batch_size), shuffle_size_(shuffle_size) {
  CHECK_GE(shuffle_size, batch_size);
  workers_.emplace_back([this] { WorkerThread(); });
  workers_.emplace_back([this] { WorkerThread(); });
}

ShufflingTrainer::~ShufflingTrainer() {
  {
    absl::MutexLock lock(&mu_);
    stopped_ = true;
  }
  for (auto& t : workers_) {
    t.join();
  }
}

void ShufflingTrainer::Train(std::unique_ptr<TrainingSample> sample) {
  absl::MutexLock lock(&mu_);

  auto has_space = [this]() -> bool { return data_.size() < max_size_; };
  mu_.Await(absl::Condition(&has_space));

#if 0
  if (b.ply() < 10) {
    // Discard half of early states to increase variance.
    if (rng_() % 2 == 0) {
      return;
    }
  }
#endif

  data_.push_back(std::move(sample));
  // TODO: Also train flipped version of the board, and rotated if there are no
  // pawns left.
  ++since_full_flush_;
  if (since_full_flush_ > 10 * shuffle_size_) {
    Flush();
    since_full_flush_ = 0;
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

  while (true) {
    std::vector<std::unique_ptr<TrainingSample>> batch_samples(batch_size_);
    {
      absl::MutexLock lock(&mu_);
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
        batch_samples[i] = std::move(data_[x]);
        std::swap(data_[x], data_.back());
        data_.pop_back();
      }
    }

    for (int i = 0; i < batch_size_; ++i) {
      const auto& sample = batch_samples[i];
      BoardToTensor(sample->board, board_tensor.SubSlice(i));

      auto move_vec = move_tensor.SubSlice(i);
      for (int i = 0; i < kMoveVectorSize; ++i) {
        move_vec.vec<float>()(i) = 0.0;
      }
      const Color turn = sample->board.turn();
      for (const auto& move : sample->moves) {
        move_vec.vec<float>()(EncodeMove(turn, move.first)) = move.second;
      }

      const double value =
          sample->winner == sample->board.turn()
              ? 1
              : sample->winner == OtherColor(sample->board.turn()) ? -1 : 0;
      value_tensor.flat<float>()(i) = value;
    }

    model_->RunTrainStep(board_tensor, move_tensor, value_tensor);
    num_trained_.fetch_add(batch_size_, std::memory_order_relaxed);

    // absl::SleepFor(absl::Milliseconds(50));
  }
}

}  // namespace chess
