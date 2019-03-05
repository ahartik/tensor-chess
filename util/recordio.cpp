#include "util/recordio.h"

#include <cstdint>
#include <cstring>

#include <iostream>

namespace util {

const uint8_t kCurrentVersion = 1;

RecordWriter::RecordWriter(const char* file) : fname_(file), out_(file) {
  if (out_) {
    out_.put(kCurrentVersion);
  }
}

RecordWriter::~RecordWriter() {
  if (!closed_) {
    if (!Finish()) {
      std::cerr << "Failed to close file " << fname_ << "\n";
      abort();
    }
  }
}

bool RecordWriter::Write(absl::string_view str) {
  const uint32_t len = str.size();
  if (!out_.write(reinterpret_cast<const char*>(&len), sizeof(len))) {
    return false;
  }
  return out_.write(str.data(), str.size()).good();
}

bool RecordWriter::Finish() {
  bool status = out_.flush().good();
  out_.close();
  closed_ = true;
  return status;
}

void RecordWriter::Flush() { out_.flush(); }

RecordReader::RecordReader(absl::string_view file)
    : in_(std::string(file).c_str()) {
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
  return in_.read(&buf[0], len).good();
}

}  // namespace util
