#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: net_snapshot_to_binary input_binary_snapshot_file output_matlab_readable_binary";
    return 0;
  }

  NetParameter net_param;
  LOG(INFO) << "Reading from " << argv[1];
  ReadProtoFromBinaryFile(argv[1], &net_param);
  int num_layers = net_param.layers_size();
  std::ofstream out_file (argv[2], std::ios::out | std::ios::binary);
  for (int i = 0; i < num_layers; ++i) {
    const LayerParameter& curr_layer = net_param.layers(i);
    const string& curr_layer_name = curr_layer.name();
    LOG(INFO) << "Layer number " << i << ": " << curr_layer_name << ". Has " << curr_layer.blobs_size() << " blobs.";
    for (int j=0; j < curr_layer.blobs_size(); ++j) {
      Blob<float> curr_blob;
      uint32_t imsize[4];
      curr_blob.FromProto(curr_layer.blobs(j));
      imsize[0] = curr_blob.num(); 
      imsize[1] = curr_blob.channels(); 
      imsize[2] = curr_blob.height(); 
      imsize[3] = curr_blob.width();
      LOG(INFO) << "  Blob " << j << ", size " << curr_blob.num() << "x" << curr_blob.channels() << "x" << curr_blob.height() << "x" << curr_blob.width() << ". First element " << curr_blob.cpu_data()[0];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(curr_blob.cpu_data()), curr_blob.count()*sizeof(float));
      if (curr_layer.blobs(j).diff_size() > 0) {
        LOG(INFO) << "  Writing the gradient";
        out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
        out_file.write(reinterpret_cast<const char*>(curr_blob.cpu_diff()), curr_blob.count()*sizeof(float));
      }
    }
  }
  
  out_file.close(); 

  return 0;
}

