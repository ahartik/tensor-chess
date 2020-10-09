#include "util/recordio.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace util {
namespace {

// Just a simple sanity test fornow.
TEST(RecordIOTest, WriteAndRead) {
  const char* path = "/tmp/test.recordio";
  RecordWriter writer(path);
  ASSERT_TRUE(writer.Write("hello"));
  ASSERT_TRUE(writer.Write("world"));
  ASSERT_TRUE(writer.Finish());

  RecordReader reader(path);
  std::string buf;
  ASSERT_TRUE(reader.Read(buf));
  EXPECT_EQ(buf, "hello");
  ASSERT_TRUE(reader.Read(buf));
  EXPECT_EQ(buf, "world");
  EXPECT_FALSE(reader.Read(buf));
}

}
}  // namespace util
