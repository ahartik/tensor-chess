#ifndef _UTIL_RECORDIO_H_
#define _UTIL_RECORDIO_H_

#include <fstream>
#include <string_view>

namespace util {

class RecordWriter {
 public:
  explicit RecordWriter(const char* file);
  bool ok() const { return (bool)out_; }
  bool Write(std::string_view str);
  bool Finish();

 private:
  std::ofstream out_;
};

class RecordReader {
 public:
  explicit RecordReader(const char* file);
  bool Read(std::string& buf);

 private:
  std::ifstream in_;
};

}  // namespace util

#endif
