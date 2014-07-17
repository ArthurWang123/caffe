// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}
  
template <typename Dtype>
Dtype caffe_rng_generate(const RandomGeneratorParameter param) {
  const std::string rand_type =  param.rand_type();
  //std::cout << rand_type << " " << rand_type.compare("uniform") << " " << rand_type.compare("gaussian") << " " << rand_type.compare("bernoulli");
  Dtype rand;
  if (rand_type.compare("uniform") == 0) {
    float tmp;
    if (param.spread() > 0.)
      caffe_rng_uniform(1, param.mean() - param.spread(), param.mean() + param.spread(), &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("gaussian") == 0) {
    float tmp;
    if (param.spread() > 0.)
      caffe_rng_gaussian(1, param.mean(), param.spread(), &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("bernoulli") == 0) {
    int tmp;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp);
    else
      tmp = 0;
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("uniform_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (param.spread() > 0.) 
      caffe_rng_uniform(1, param.mean() - param.spread(), param.mean() + param.spread(), &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else if (rand_type.compare("gaussian_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (param.spread() > 0.) 
      caffe_rng_gaussian(1, param.mean(), param.spread(), &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else {
    LOG(ERROR) << "Unknown random type " << rand_type;
    rand = NAN;
  }
  return rand;
}  


template <typename Dtype>
void DataAugmentationLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Data aumentation layer takes one input blob.";
  CHECK_GE(top->size(), 1) << "Data Layer takes one or two output blobs.";
  CHECK_LE(top->size(), 2) << "Data Layer takes one or two output blobs.";

  if (top->size() == 1) {
    output_params_ = false;
  } else {
    output_params_ = true;
  }
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
//   AugmentationParameter augmentation_param = this->layer_param_.augmentation_param();
//   
//   if (augmentation_param.has_rotate()) {
//     std::cout << "Rotation: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<float>(augmentation_param.rotate()) << std::endl;
//   }
//   if (augmentation_param.has_translate()) {
//     std::cout << "Translation: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<float>(augmentation_param.translate()) << std::endl;
//   }
//   if (augmentation_param.has_mirror()) {
//     std::cout << "Mirror: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<bool>(augmentation_param.mirror()) << std::endl;
//   }  
//   std::cin.get();

  //num pixels to crop left/right and top/bottom
  int crop_size = this->layer_param_.augmentation_param().crop_size();
  CHECK_GE(height, crop_size) << "crop size greater than original";
  CHECK_GE(width, crop_size) << "crop size greater than original";
  
  cropped_height_ = crop_size;
  cropped_width_ = crop_size;

  (*top)[0]->Reshape(num, channels, crop_size, crop_size);
  
  if (output_params_) {
    num_params_ = 35;
    (*top)[1]->Reshape(num, num_params_, 1, 1);
  }
  
  if (this->layer_param_.augmentation_param().recompute_mean()) {
    data_mean_.Reshape(1, channels, crop_size, crop_size);
  }
  
  num_iter_ = 0;
  
}

template <typename Dtype>
Dtype DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  
  num_iter_++;

  AugmentationParameter aug = this->layer_param_.augmentation_param();
  if (!aug.has_crop_size())
    LOG(ERROR) << "Please enter crop_size if you want to perform augmentation";
  const int crop_size = aug.crop_size();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();  
  
  std::string write_augmented;
  if (aug.has_write_augmented())
    write_augmented = aug.write_augmented();
  else
    write_augmented = std::string("");
  
  bool augment_during_test = aug.augment_during_test();  
  bool train_phase = (Caffe::phase() == Caffe::TRAIN);
  //LOG(INFO) <<  " === train_phase " << train_phase;
  
#pragma omp parallel for firstprivate(aug, train_phase, write_augmented, augment_during_test)
  for (int item_id = 0; item_id < num; ++item_id) {
    
    int x, y, c, top_idx, bottom_idx, h_off, w_off;
    float x1, y1, x2, y2;
    
    bool do_spatial_transform, do_chromatic_transform;
    
    bool do_rotate = aug.has_rotate();
    bool do_translate = aug.has_translate();
    bool do_mirror = aug.has_mirror();
    bool do_zoom = aug.has_zoom();
    
    bool do_pow_nomean [3] = {false, false, false};
    bool do_mult_nomean [3] = {false, false, false};
    bool do_add_nomean [3] = {false, false, false};
    bool do_pow_withmean [3] = {false, false, false};
    bool do_mult_withmean [3] = {false, false, false};
    bool do_add_withmean [3] = {false, false, false};
    bool do_lmult_pow = aug.has_lmult_pow();
    bool do_lmult_add = aug.has_lmult_add();
    bool do_lmult_mult = aug.has_lmult_mult();
    bool do_col_rotate = aug.has_col_rotate();
    
    Dtype angle;
    Dtype zoom_coeff;
    Dtype dx;
    Dtype dy;
    bool mirror;  
    
    Dtype lmult_pow_coeff;
    Dtype lmult_mult_coeff;
    Dtype lmult_add_coeff;  
    Dtype pow_coeffs_nomean [3];
    Dtype mult_coeffs_nomean [3];
    Dtype add_coeffs_nomean [3];
    Dtype pow_coeffs_withmean [3];
    Dtype mult_coeffs_withmean [3];
    Dtype add_coeffs_withmean [3];
    Dtype col_angle;
    
    if (output_params_) {
      Dtype* top_params = (*top)[1]->mutable_cpu_data();
      for (int i=0; i<num_params_; i++)
        top_params[item_id * num_params_ + i] = 0.;
    }
    
    //   We only do transformations during training.
    if (!(train_phase || augment_during_test)) {
      do_spatial_transform   = false;
      do_chromatic_transform = false;
    }
    else {
      do_spatial_transform   = (do_mirror           || do_translate         || do_rotate           || do_zoom);
      do_chromatic_transform = (do_lmult_pow        || do_lmult_mult        || do_lmult_add        || 
                                aug.has_sat_pow()   || aug.has_sat_mult()   || aug.has_sat_add()   ||
                                aug.has_col_pow()   || aug.has_col_mult()   || aug.has_col_add()   ||
                                aug.has_ladd_pow()  || aug.has_ladd_mult()  || aug.has_ladd_add()  || do_col_rotate);
      angle = 0.;
      zoom_coeff = 1.;
      dx = 0.;
      dy = 0.;
      mirror = false;  
      
      lmult_pow_coeff = 1.;
      lmult_mult_coeff = 1.;
      lmult_add_coeff = 0.;  
      pow_coeffs_nomean[0] = 1.;  pow_coeffs_nomean[1] = 1.;  pow_coeffs_nomean[2] = 1.; 
      mult_coeffs_nomean[0] = 1.; mult_coeffs_nomean[1] = 1.; mult_coeffs_nomean[2] = 1.; 
      add_coeffs_nomean[0] = 0.;  add_coeffs_nomean[1] = 0.;  add_coeffs_nomean[2] = 0.; 
      pow_coeffs_withmean[0] = 1.;  pow_coeffs_withmean[1] = 1.;  pow_coeffs_withmean[2] = 1.; 
      mult_coeffs_withmean[0] = 1.; mult_coeffs_withmean[1] = 1.; mult_coeffs_withmean[2] = 1.; 
      add_coeffs_withmean[0] = 0.;  add_coeffs_withmean[1] = 0.;  add_coeffs_withmean[2] = 0.;
      col_angle = 0.;
    }
      
    

    
    //LOG(INFO) <<  " === thread " << omp_get_thread_num() << "/" << omp_get_num_threads() << " === ";
    

    
    // sample the parameters of the transoformations  
    if (do_spatial_transform) {
      int counter = 0;
      int max_num_tries = 20;    
      int good_params = 0;
      
      // try to sample parameters for which transformed image doesn't go outside the borders of the original one
      // in order to do check this, just apply the transformations to 4 corners
      while (good_params < 4 && counter < max_num_tries) {
        good_params = 0;
        if (aug.has_rotate())
          angle = caffe_rng_generate<float>(aug.rotate());
        if (aug.has_zoom())
          zoom_coeff = caffe_rng_generate<float>(aug.zoom());
        if (aug.has_translate()) {
          dx = caffe_rng_generate<float>(aug.translate());
          dy = caffe_rng_generate<float>(aug.translate());
        }
        if (aug.has_mirror())
          mirror = caffe_rng_generate<bool>(aug.mirror());
  
        //LOG(INFO) << "angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
        
        for (x = 0; x < crop_size; x += crop_size-1) {
          for (y = 0; y < crop_size; y += crop_size-1) {
            // move the origin and mirror
            if (mirror) {
              x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
              y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(crop_size);            
            } 
            else {
              x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
              y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(crop_size);
            }
            // rotate
            x2 =  cos(angle) * x1 - sin(angle) * y1;
            y2 =  sin(angle) * x1 + cos(angle) * y1;
            // translate
            x2 = x2 + dx * static_cast<Dtype>(crop_size);
            y2 = y2 + dy * static_cast<Dtype>(crop_size);
            // zoom
            x2 = x2 / zoom_coeff;
            y2 = y2 / zoom_coeff;
            // move the origin back
            x2 = x2 + .5 * static_cast<Dtype>(width);
            y2 = y2 + .5 * static_cast<Dtype>(height);
            
            if (!(floor(x2) < 0 || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0 || floor(y2) > static_cast<Dtype>(height - 2)))
                good_params++;
            //mexPrintf(" (%f,%f) ", x2, y2);
          }
        }
        //mexPrintf("\n");
        counter++;
      }
      if (counter >= max_num_tries) {
        angle=0.;
        zoom_coeff=1.;
        dx=0.;
        dy=0.;
        mirror = false;
      } 
      
      if (do_rotate)
        do_rotate = (fabs(angle) >1e-2);
      if (do_translate)
        do_translate = ( fabs(dx) > 1e-2 || fabs(dy) > 1e-2) ;
      if (do_mirror)
        do_mirror = mirror;
      if (do_zoom)
        do_zoom = (fabs(zoom_coeff - 1.) >1e-2);
      
      do_spatial_transform = (do_rotate || do_translate || do_mirror || do_zoom);
      
      if (write_augmented.size()) { 
        if (do_spatial_transform)
          LOG(INFO) << "Augmenting " << item_id << ". angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
        else
          LOG(INFO) << "Not augmenting " << item_id << " spatially";
      } 
      
      if (output_params_) {
        Dtype* top_params = (*top)[1]->mutable_cpu_data();
        if (do_mirror)
          top_params[item_id * num_params_ + 0] = 1.;
        if (do_translate) {
          top_params[item_id * num_params_ + 1] = dx;
          top_params[item_id * num_params_ + 2] = dy;
        }
        if (do_rotate)
          top_params[item_id * num_params_ + 3] = angle;
        if (do_zoom)
          top_params[item_id * num_params_ + 4] = log(zoom_coeff);
      }
    } 
    
    if (do_chromatic_transform) {
      if (aug.has_lmult_pow())
        lmult_pow_coeff = caffe_rng_generate<float>(aug.lmult_pow());
      if (aug.has_lmult_mult())
        lmult_mult_coeff = caffe_rng_generate<float>(aug.lmult_mult());
      if (aug.has_lmult_add())
        lmult_add_coeff = caffe_rng_generate<float>(aug.lmult_add());
      
      if (aug.has_ladd_pow())
        pow_coeffs_nomean[0] = caffe_rng_generate<float>(aug.ladd_pow());
      if (aug.has_ladd_mult())
        mult_coeffs_nomean[0] = caffe_rng_generate<float>(aug.ladd_mult());
      if (aug.has_ladd_add())
        add_coeffs_nomean[0] = caffe_rng_generate<float>(aug.ladd_add());      

      if (aug.has_sat_pow()) {
        pow_coeffs_withmean[1] = caffe_rng_generate<float>(aug.sat_pow());
        pow_coeffs_withmean[2] = pow_coeffs_withmean[1];
      }
      if (aug.has_sat_mult()) {
        mult_coeffs_withmean[1] = caffe_rng_generate<float>(aug.sat_mult()); 
        mult_coeffs_withmean[2] = mult_coeffs_withmean[1];
      }
      if (aug.has_sat_add()) {
        add_coeffs_withmean[1] = caffe_rng_generate<float>(aug.sat_add());
        add_coeffs_withmean[2] = add_coeffs_withmean[1];
      }
      
      for (c=1; c<=2; c++) {
        if (aug.has_col_pow())
          pow_coeffs_nomean[c] = caffe_rng_generate<float>(aug.col_pow());
        if (aug.has_col_mult())
          mult_coeffs_nomean[c] = caffe_rng_generate<float>(aug.col_mult());
        if (aug.has_col_add())
          add_coeffs_nomean[c] = caffe_rng_generate<float>(aug.col_add());
      }
      
      if (aug.has_col_rotate())
        col_angle = caffe_rng_generate<float>(aug.col_rotate());
                  
      do_chromatic_transform = false;
    
      for (c=0; c<3; c++) {
        do_pow_nomean[c] = (fabs(pow_coeffs_nomean[c] - 1.) > 1e-2);
        do_add_nomean[c] = (fabs(add_coeffs_nomean[c]) > 1e-2);
        do_mult_nomean[c] = (fabs(mult_coeffs_nomean[c] - 1.) > 1e-2);
        do_pow_withmean[c] = (fabs(pow_coeffs_withmean[c] - 1.) > 1e-2);
        do_add_withmean[c] = (fabs(add_coeffs_withmean[c]) > 1e-2);
        do_mult_withmean[c] = (fabs(mult_coeffs_withmean[c] - 1.) > 1e-2);
          
        do_chromatic_transform = (do_chromatic_transform || do_pow_nomean[c] || do_add_nomean[c] || do_mult_nomean[c] || do_pow_withmean[c] || do_add_withmean[c] || do_mult_withmean[c]);
      }
      if (do_lmult_pow)
        do_lmult_pow = (fabs(lmult_pow_coeff - 1.) > 1e-2);
      if (do_lmult_add)
        do_lmult_add = (fabs(lmult_add_coeff) > 1e-2);
      if (do_lmult_mult)
        do_lmult_mult = (fabs(lmult_mult_coeff - 1.) > 1e-2);
      if (do_col_rotate)
        do_col_rotate = (fabs(col_angle) > 1e-2);
      do_chromatic_transform = (do_chromatic_transform || do_lmult_pow || do_lmult_add || do_lmult_mult || do_col_rotate);
      
      if (write_augmented.size()) {
        if (do_chromatic_transform)
          LOG(INFO) << "Augmenting " << item_id << ". lmult_pow: " << lmult_pow_coeff << ", lmult_mult: " << lmult_mult_coeff << ", lmult_add: " << lmult_add_coeff
          << ", pow_nm[0]: " << pow_coeffs_nomean[0]    << ", mult_nm[0]: " << mult_coeffs_nomean[0]    << ", add_nm[0]: " << add_coeffs_nomean[0]
          << ", pow_nm[1]: " << pow_coeffs_nomean[1]    << ", mult_nm[1]: " << mult_coeffs_nomean[1]    << ", add_nm[1]: " << add_coeffs_nomean[1]
          << ", pow_nm[2]: " << pow_coeffs_nomean[2]    << ", mult_nm[2]: " << mult_coeffs_nomean[2]    << ", add_nm[2]: " << add_coeffs_nomean[2]
          << ", pow_wm[0]: " << pow_coeffs_withmean[0]    << ", mult_wm[0]: " << mult_coeffs_withmean[0]    << ", add_wm[0]: " << add_coeffs_withmean[0]
          << ", pow_wm[1]: " << pow_coeffs_withmean[1]    << ", mult_wm[1]: " << mult_coeffs_withmean[1]    << ", add_wm[1]: " << add_coeffs_withmean[1]
          << ", pow_wm[2]: " << pow_coeffs_withmean[2]    << ", mult_wm[2]: " << mult_coeffs_withmean[2]    << ", add_wm[2]: " << add_coeffs_withmean[2]
          << ", col_angle: " << col_angle;
        else
          LOG(INFO) << "Not augmenting " << item_id << " chromativally";
      }
      
      if (output_params_) {
        Dtype* top_params = (*top)[1]->mutable_cpu_data();
        top_params[item_id * num_params_ + 10] = log(lmult_pow_coeff);
        top_params[item_id * num_params_ + 11] = lmult_add_coeff;
        top_params[item_id * num_params_ + 12] = log(lmult_mult_coeff);
        top_params[item_id * num_params_ + 13] = log(pow_coeffs_nomean[0]);
        top_params[item_id * num_params_ + 14] = add_coeffs_nomean[0];
        top_params[item_id * num_params_ + 15] = log(mult_coeffs_nomean[0]);
        top_params[item_id * num_params_ + 16] = log(pow_coeffs_nomean[1]);
        top_params[item_id * num_params_ + 17] = add_coeffs_nomean[1];
        top_params[item_id * num_params_ + 18] = log(mult_coeffs_nomean[1]);
        top_params[item_id * num_params_ + 19] = log(pow_coeffs_nomean[2]);
        top_params[item_id * num_params_ + 20] = add_coeffs_nomean[2];
        top_params[item_id * num_params_ + 21] = log(mult_coeffs_nomean[2]);
        top_params[item_id * num_params_ + 22] = log(pow_coeffs_withmean[0]);
        top_params[item_id * num_params_ + 23] = add_coeffs_withmean[0];
        top_params[item_id * num_params_ + 24] = log(mult_coeffs_withmean[0]);
        top_params[item_id * num_params_ + 25] = log(pow_coeffs_withmean[1]);
        top_params[item_id * num_params_ + 26] = add_coeffs_withmean[1];
        top_params[item_id * num_params_ + 27] = log(mult_coeffs_withmean[1]);
        top_params[item_id * num_params_ + 28] = log(pow_coeffs_withmean[2]);
        top_params[item_id * num_params_ + 29] = add_coeffs_withmean[2];
        top_params[item_id * num_params_ + 30] = log(mult_coeffs_withmean[2]);
        top_params[item_id * num_params_ + 31] = col_angle;       
      }
    }      
    
    
    // actually apply the transformation
    if (do_spatial_transform) { 
      int i00,i01,i10,i11;
      for (x = 0; x < crop_size; x++) {
        for (y = 0; y < crop_size; y++) {
          // move the origin and mirror
          if (mirror) {
            x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
            y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(crop_size);            
          } 
          else {
            x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
            y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(crop_size);
          }
          // rotate
          if (do_rotate) {
            x2 =  cos(angle) * x1 - sin(angle) * y1;
            y2 =  sin(angle) * x1 + cos(angle) * y1;
          }
          else {
            x2 = x1;
            y2 = y1;
          }
          // translate
          if (do_translate) {
            x2 = x2 + dx * static_cast<Dtype>(crop_size);
            y2 = y2 + dy * static_cast<Dtype>(crop_size);
          }
          // zoom
          if (do_zoom) {
            x2 = x2 / zoom_coeff;
            y2 = y2 / zoom_coeff;
          }
          // move the origin back
          x2 = x2 + .5 * static_cast<Dtype>(width);
          y2 = y2 + .5 * static_cast<Dtype>(height);
          

          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*crop_size + x)*crop_size + y;
            if (floor(x2) < 0. || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0. || floor(y2) > static_cast<Dtype>(height - 2))
              top_data[top_idx] = 0.;
            else {
              if (do_rotate || do_zoom) {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));
                i01 = i00 + 1;
                i10 = i00 + height;
                i11 = i00 + height + 1;
                
                top_data[top_idx] = bottom_data[i00] * ((floor(x2)+1)  - x2) * ((floor(y2)+1)  - y2) +
                                    bottom_data[i01] * ((floor(x2)+1)  - x2) * (y2 - floor(y2))      +
                                    bottom_data[i10] * (x2 - floor(x2))      * ((floor(y2)+1)  - y2) +
                                    bottom_data[i11] * (x2 - floor(x2))      * (y2 - floor(y2));                
              } 
              else {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));              
                top_data[top_idx] = bottom_data[i00];
              }
            }         
            // TODO: return the mean when end debugging
            //top_data[i] = (top_data[i] - 127.5) * scale;
          }
          //mexPrintf(" (%f,%f) ", x2, y2);        
        }
      }
    }
    else {
      h_off = (height - crop_size)/2;
      w_off = (width - crop_size)/2;
      for (x = 0; x < crop_size; x++) {
        for (y = 0; y < crop_size; y++) {
          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*crop_size + x)*crop_size + y;
            bottom_idx = ((item_id*channels + c)*width + x + w_off)*height + y + h_off;
            top_data[top_idx] = bottom_data[bottom_idx];
          }
        }
      }
    }
    
    if (do_chromatic_transform) {
//       LOG(INFO) << " >>> do chromatic transform " << item_id;
      Dtype s, s1, l, l1, max_l;
      Dtype rgb [3];
      Dtype eig [3];
      Dtype mean_eig [3];
      Dtype max_abs_eig[3] = {0., 0., 0.};
      Dtype max_rgb[3] = {0., 0., 0.};
      Dtype min_rgb[3] = {0., 0., 0.};
      Dtype mean_rgb[3] = {0., 0., 0.};
//      const Dtype eigvec [9] = {0.57, 0.58, 0.57, -0.72, 0.03, 0.68, -0.38, 0.81, -0.44};
      const Dtype eigvec [9] = {0.5579, 0.5859, 0.5878, 0.8021, -0.1989, -0.5631, -0.2130, 0.7856, -0.5809};
//       const Dtype eigvec [9] = {0.5878, 0.5859, 0.5579, -0.5631, -0.1989, 0.8021, -0.5809, 0.7856, -0.2130};
      // compute max abs values of eigs (projections onto color space eigenvectors)
      for (x=0; x<crop_size; x++) {
        for (y=0; y<crop_size; y++) {
          for (c=0; c<channels; c++)
            rgb[c] = top_data[((item_id*channels + c)*crop_size + x)*crop_size + y];
          for (c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if (fabs(eig[c]) > max_abs_eig[c])
              max_abs_eig[c] = fabs(eig[c]);
            if (rgb[c] > max_rgb[c])
              max_rgb[c] = rgb[c];
            if (rgb[c] < min_rgb[c])
              min_rgb[c] = rgb[c];
            mean_rgb[c] = mean_rgb[c] + rgb[c]/crop_size/crop_size;
          }
        }
      }
      max_l = sqrt(max_abs_eig[0]*max_abs_eig[0] + max_abs_eig[1]*max_abs_eig[1] + max_abs_eig[2]*max_abs_eig[2]);;
        
      // actually apply the transform
      for (c=0; c<channels; c++) 
        mean_eig[c] = eigvec[3*c] * mean_rgb[0] + eigvec[3*c+1] * mean_rgb[1] + eigvec[3*c+2] * mean_rgb[2];
        
      for (x=0; x<crop_size; x++) {
        for (y=0; y<crop_size; y++) {
          // subtracting the mean
          for (c=0; c<channels; c++) {
            rgb[c] = top_data[((item_id*channels + c)*crop_size + x)*crop_size + y];
            rgb[c] = rgb[c] - mean_rgb[c];
          }
          // doing the nomean stuff
          for (c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if ( max_abs_eig[c] > 1e-2 ) {
              eig[c] = eig[c] / max_abs_eig[c]; 
              if (do_pow_nomean[c])            
                eig[c] = static_cast<float>(sgn(eig[c])) * pow(fabs(eig[c]), pow_coeffs_nomean[c]);
              if (do_add_nomean[c])                 
                eig[c] = eig[c] + add_coeffs_nomean[c];
              if (do_mult_nomean[c])
                eig[c] = eig[c] * mult_coeffs_nomean[c];
            }
          }
          // re-adding the mean
          for (c=0; c<channels; c++)
            eig[c] = eig[c] + mean_eig[c] / max_abs_eig[c];
          // doing the withmean stuff
          if ( max_abs_eig[c] > 1e-2 && (do_pow_withmean[0] || do_add_withmean[0] || do_mult_withmean[0])) {
            if (do_pow_withmean[0])            
              eig[0] = static_cast<float>(sgn(eig[0])) * pow(fabs(eig[0]), pow_coeffs_withmean[0]);
            if (do_add_withmean[0])                 
              eig[0] = eig[0] + add_coeffs_withmean[0];
            if (do_mult_withmean[0])
              eig[0] = eig[0] * mult_coeffs_withmean[0];
          }
          if (do_pow_withmean[1] || do_add_withmean[1] || do_mult_withmean[1]) {
            s = sqrt(eig[1]*eig[1] + eig[2]*eig[2]);
            s1 = s;
            if (s > 1e-2) {
              if (do_pow_withmean[1])            
                s1 = pow(s1, pow_coeffs_withmean[1]);
              if (do_add_withmean[1])                 
                s1 = fmax(s1 + add_coeffs_withmean[1], 0.);
              if (do_mult_withmean[1])
                s1 = s1 * mult_coeffs_withmean[1];              
            }
          }
          if (do_col_rotate) {
            Dtype temp1, temp2;
            temp1 =  cos(col_angle) * eig[1] - sin(col_angle) * eig[2];
            temp2 =  sin(col_angle) * eig[1] + cos(col_angle) * eig[2]; 
            eig[1] = temp1;
            eig[2] = temp2;
          }
          for (c=0; c<channels; c++) {
            if ( max_abs_eig[c] > 1e-2 ) {
              eig[c] = eig[c] * max_abs_eig[c]; 
            }
          }
          if (max_l > 1e-2 && (do_lmult_pow || do_lmult_add || do_lmult_mult) || (do_pow_withmean[1] || do_add_withmean[1] || do_mult_withmean[1])) {
            l1 = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            l1 = l1 / max_l;
          }
          if (s > 1e-2 && (do_pow_withmean[1] || do_add_withmean[1] || do_mult_withmean[1])) {
            eig[1] = eig[1] / s * s1;
            eig[2] = eig[2] / s * s1;
          }
          if ( max_l > 1e-2 && (do_lmult_pow || do_lmult_add || do_lmult_mult) || (do_pow_withmean[1] || do_add_withmean[1] || do_mult_withmean[1])) {            
            l = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            if (do_lmult_pow)
              l1 = pow(l1, lmult_pow_coeff);
            if (do_lmult_add)
              l1 = fmax(l1 + lmult_add_coeff, 0.);
            if (do_lmult_mult)
              l1 = l1 * lmult_mult_coeff;
            l1 = l1 * max_l;
            if (l > 1e-2)
              for (c=0; c<channels; c++) {
                eig[c] = eig[c] / l * l1;
                if (eig[c] > max_abs_eig[c])
                  eig[c] = max_abs_eig[c];
              }
          }                             
          for (c=0; c<channels; c++) {
            rgb[c] = eigvec[c] * eig[0] + eigvec[3+c] * eig[1] + eigvec[6+c] * eig[2];
            if (rgb[c] > aug.max_multiplier()*max_rgb[c])
              rgb[c] = aug.max_multiplier()*max_rgb[c];
            if (rgb[c] < aug.max_multiplier()*min_rgb[c])
              rgb[c] = aug.max_multiplier()*min_rgb[c]; 
            top_data[((item_id*channels + c)*crop_size + x)*crop_size + y] = rgb[c];
          }          
        }
      } 
    }    
  }
  
  if(aug.recompute_mean() > 0 && num_iter_ <= aug.recompute_mean() ) {
    Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
    int count = crop_size*crop_size*channels;    
    for (int c = 0; c < count; ++c) {
      data_mean_cpu[c] = data_mean_cpu[c]*(static_cast<Dtype>(num_iter_)-1);
      for (int item_id = 0; item_id < num; ++item_id) 
        data_mean_cpu[c] = data_mean_cpu[c] + top_data[item_id*count + c] / num;
      data_mean_cpu[c] = data_mean_cpu[c] / static_cast<Dtype>(num_iter_);      
    }  
  }
    
  if(aug.recompute_mean() > 0) {
    Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
    int count = crop_size*crop_size*channels;   
    for (int item_id = 0; item_id < num; ++item_id) {
      for (int c = 0; c < count; ++c) {
        top_data[item_id*count + c] = top_data[item_id*count + c] - data_mean_cpu[c];
      }
    }    
  }
  
  if (write_augmented.size()) {  
    std::ofstream out_file (write_augmented.data(), std::ios::out | std::ios::binary);
    if (out_file.is_open()) { 
      uint32_t imsize[4];
      imsize[0] = num; 
      imsize[1] = channels; 
      imsize[2] = crop_size; 
      imsize[3] = crop_size;
      LOG(INFO) << "Writing blob size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(top_data), imsize[0]*imsize[1]*imsize[2]*imsize[3]*sizeof(float));
      out_file.close();
      LOG(INFO) << " finished augmenting a batch. train=" << train_phase << " === PAUSED === ";
      std::cout << " finished augmenting a batch. train=" << train_phase << " === PAUSED === ";
      std::cin.get();
    }
    else
      LOG(INFO) << "WARNING: Could not open the file" << write_augmented;
  }
  
  if (aug.write_mean().size()) {  
    std::ofstream out_file (aug.write_mean().data(), std::ios::out | std::ios::binary);
    if (out_file.is_open()) { 
      uint32_t imsize[4];
      imsize[0] = 1; 
      imsize[1] = channels; 
      imsize[2] = crop_size; 
      imsize[3] = crop_size;
      Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
      LOG(INFO) << "Writing blob size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(data_mean_cpu), imsize[0]*imsize[1]*imsize[2]*imsize[3]*sizeof(float));
      out_file.close();
      LOG(INFO) << " finished writing the mean. num_iter_=" << num_iter_ << " === PAUSED === ";
      std::cout << " finished writing the mean. num_iter_=" << num_iter_ << " === PAUSED === ";
      std::cin.get();
    }
    else
      LOG(INFO) << "WARNING: Could not open the file" << write_augmented;
  }
  
  
 
  
//   for (int item_id = 0; item_id < num; ++item_id) {
//     for (int c = 0; c < channels; ++c) {
//       for (int h = 0; h < this->cropped_height; ++h) {
//         for (int w = 0; w < this->cropped_width; ++w)  {
//           int bottom_idx;
//           if (mirror) {
//             bottom_idx = item_id * (channels * width * height) + c * (height*width)
//               + (h_off+h) * width + width - 1 - w_off - w;
//           } else {
//             bottom_idx = item_id * (channels * width * height) + c * (height*width)
//               + (h_off+h)*width + (w_off + w);
//           }
//           int top_idx = item_id * (channels * this->cropped_height * this->cropped_width)
//             + c * (this->cropped_height * this->cropped_width) + h * this->cropped_width + w;
//           top_data[top_idx] = bottom_data[bottom_idx];
//         }
//       }
//     }
//   }

  return Dtype(0);
}


INSTANTIATE_CLASS(DataAugmentationLayer);


}  // namespace caffe
