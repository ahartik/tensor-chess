#ifndef _UTIL_RECORDIO_H_
#define _UTIL_RECORDIO_H_

#include <fstream>
#include <string>

#include "absl/strings/string_view.h"

namespace util {

class RecordWriter {
 public:
  explicit RecordWriter(const char* file);
  ~RecordWriter();

  bool ok() const { return !closed_ && (bool)out_; }
  bool Write(absl::string_view str);
  bool Finish();
  void Flush();

 private:
  std::string fname_;
  std::ofstream out_;
  bool closed_ = false;
};

class RecordReader {
 public:
  explicit RecordReader(absl::string_view file);
  bool Read(std::string& buf);

 private:
  std::ifstream in_;
};

}  // namespace util

#endif
