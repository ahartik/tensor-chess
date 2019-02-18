#include "util/recordio.h"

#include <cstring>
#include <cstdint>

#include <iostream>

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
  return out_.write(str.data(), str.size()).good();
}

bool RecordWriter::Finish() {
  bool status = out_.flush().good();
  out_.close();
  return status;
}

RecordReader::RecordReader(const char* file) : in_(file) {
  const uint8_t version = in_.get();
  if (version != kCurrentVersion) {
    std::cerr << "Unsupported recordio version " << version << "file: " << file
              << "\n";
    abort();
  }
}

bool RecordReader::Read(std::string& buf) {
  uint32_t len = 0;
  char len_buf[sizeof(len)];
  if (!in_.read(len_buf, sizeof(len))) {
    return false;
  }
  memcpy(&len, len_buf, sizeof(len));
  buf.resize(len);
  return in_.read(buf.data(), len).good();
}

}  // namespace util
