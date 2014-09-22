#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <fstream>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace caffe;
using google::protobuf::Message;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: proto_text_to_binary input_text_file output_binary_file";
    return 0;
  }

  NetParameter net_param;
  LOG(INFO) << "Converting from " << argv[1] << " to " << argv[2];
  ReadProtoFromBinaryFile(argv[1], &net_param);
  WriteProtoToTextFile(net_param, argv[2]);

  return 0;
}

