#include <iostream>

#include "util/recordio.h"

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "usage: " << argv[0] << " file1 file2\n";
    return 1;
  }
  std::string buf;
  for (int i = 1; i < argc; ++i) {
    util::RecordReader r(argv[i]);
    int count = 0;
    while (r.Read(buf)) {
      ++count;
    }
    std::cout << argv[i] << " - " << count << " records\n";
  }
  return 0;
}
