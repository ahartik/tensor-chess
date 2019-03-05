#include "absl/debugging/failure_signal_handler.h"
#include "absl/debugging/symbolize.h"

#include <cstdio>

void NiceInit(int argc, const char** argv) {
  absl::InitializeSymbolizer(argv[0]);
  absl::FailureSignalHandlerOptions opts;
  absl::InstallFailureSignalHandler(opts);
}
