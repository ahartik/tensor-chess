#include "util/recordio.h"

#include <cstdint>

namespace util {

const uint8_t kCurrentVersion = 1;

RecordWriter::RecordWriter(const char* file) : out_(file) {
  if (out_) {
    out_.put(kCurrentVersion);
  }
}

bool RecordWriter::Write(std::string_view str) {
  const uint32_t len = str.size();
  if (!out_.write(reinterpret_cast<const char*>(&len), sizeof(len))) {
    return false;
  }
  return (bool)out_.write(str.data(), str.size());
}

bool RecordWriter::Finish() { return (bool)out_.flush(); }

}  // namespace util
