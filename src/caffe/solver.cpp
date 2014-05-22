// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <ctime>
#include <cstdlib> 

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
  SolverParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  param_ = param;
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  if (param_.has_train_net_param()) {
    CHECK(!param_.has_train_net()) << "Either train_net_param or train_net may "
                                   << "be specified, but not both.";
    LOG(INFO) << "Creating training net specified in SolverParameter.";
    net_.reset(new Net<Dtype>(param_.train_net_param()));
  } else {
    LOG(INFO) << "Creating training net from file: " << param_.train_net();
    net_.reset(new Net<Dtype>(param_.train_net()));
  }
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_test_nets) {
    CHECK_EQ(param_.test_iter().size(), num_test_nets) << "you need to specify test_iter for each test network.";
    CHECK_GT(param_.test_interval(), 0);
  }
  test_nets_.resize(num_test_nets);
  for (int i = 0; i < num_test_net_params; ++i) {
      LOG(INFO) << "Creating testing net (#" << i
                << ") specified in SolverParameter.";
      test_nets_[i].reset(new Net<Dtype>(param_.test_net_param(i)));
  }
  for (int i = 0, test_net_id = num_test_net_params;
       i < num_test_net_files; ++i, ++test_net_id) {
      LOG(INFO) << "Creating testing net (#" << test_net_id
                << ") from file: " << param.test_net(i);
      test_nets_[test_net_id].reset(new Net<Dtype>(param_.test_net(i)));
  }
  CHECK_GT(this->param_.termination_criterion().size(), 0) << "at least one termination criterion needed.";
  termination_criterions_.resize(this->param_.termination_criterion().size());
  for(int i=0; i < this->param_.termination_criterion().size(); i++) {
    if (this->param_.termination_criterion().Get(i) == SolverParameter::MAX_ITER) {
      termination_criterions_[i].reset(new MaxIterTerminationCriterion<Dtype >(param_.max_iter()));
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::TEST_ACCURACY) {
      CHECK(num_test_nets) << "Test network needed for TestAccuracyTerminationCriterion.";
      bool valid_net = false;
      for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
        if (test_nets_[test_net_id]->name() == "valid") {
          valid_net = true;
        }
      }
      CHECK(valid_net) << "Network with the name 'valid' needed for TestAccuracyTerminationCriterion.";
      termination_criterions_[i].reset(new TestAccuracyTerminationCriterion<Dtype >(param_.test_accuracy_stop_countdown()));
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::EXTERNAL) {
      CHECK(param_.has_external_term_criterion_cmd()) << "external_term_criterion_cmd needed";
      CHECK(param_.has_external_term_criterion_num_iter()) << "external_term_criterion_num_iter needed";
      termination_criterions_[i].reset(new ExternalTerminationCriterion<Dtype >(
        param_.external_term_criterion_cmd(),
        param_.external_term_criterion_num_iter()
        ));
    }
  }
  LOG(INFO) << "Solver scaffolding done.";
}


template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
  if (param_.solver_mode() == SolverParameter_SolverMode_GPU &&
      param_.has_device_id()) {
    Caffe::SetDevice(param_.device_id());
  }
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << net_->name();
  PreSolve();

  iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // Run a test pass before doing any training to avoid waiting a potentially
  // very long time (param_.test_interval() training iterations) to report that
  // there's not enough memory to run the test net and crash, etc.; and to gauge
  // the effect of the first training iterations.
  if (param_.test_interval()) {
    TestAll();
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  do {
    iter_++;
    for (int i=0; i < termination_criterions_.size(); i++) {
      termination_criterions_[i]->NotifyIteration(iter_);
    }

    Dtype loss = net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    net_->Update();

    if (param_.display() && iter_ % param_.display() == 0) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      TestAll();
    }
    // Check if we need to do snapshot
    if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
  } while (!TerminationCriterionsMet());
  if (param_.snapshot_on_exit()) {
    iter_--;
    Snapshot();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
bool Solver<Dtype>::TerminationCriterionsMet() {
  for (int i=0; i < termination_criterions_.size(); i++) {
    if (termination_criterions_[i]->IsCriterionMet()) {
      return true;
    }
  }
  return false;
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  time_t timer;
  timer = time(NULL);
  LOG(INFO) << "Test timestamp " << timer;
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}


template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_ << ", Testing net (#" << test_net_id << ")";
  // We need to set phase to test before running.
  Caffe::set_phase(Caffe::TEST);
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter().Get(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_nets_[test_net_id]->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter().Get(test_net_id);
    LOG(INFO) << test_nets_[test_net_id]->name() << " test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    LOG(INFO) << test_nets_[test_net_id]->name() << " test score #" << i << ": "
        << test_score[i] / param_.test_iter().Get(test_net_id);
  }
  if (test_nets_[test_net_id]->name() == "valid") {
    double valid_accuracy = test_score[0] / param_.test_iter().Get(test_net_id);
    for (int i=0; i < termination_criterions_.size(); i++) {
      termination_criterions_[i]->NotifyTestAccuracy(valid_accuracy);
    }
  }
  Caffe::set_phase(Caffe::TRAIN);
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  LOG(INFO) << "Snapshotting to " << filename;
  WriteProtoToBinaryFile(net_param, filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(filename);
  filename += ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << filename;
  WriteProtoToBinaryFile(state, filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    CHECK_GT(this->param_.stepsize(), 0) << "step size necessary.";
    int current_step = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), current_step);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "inv_bergstra_bengio") {
    CHECK_GT(this->param_.stepsize(), 0) << "step size necessary.";
    rate = (this->iter_ > this->param_.stepsize()) ? this->param_.base_lr() * Dtype(this->param_.stepsize()) / this->iter_
      : this->param_.base_lr();
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}


template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
      if (local_decay) {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->cpu_data(),
            history_[param_id]->mutable_cpu_data());
      }
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->gpu_diff(), momentum,
          history_[param_id]->mutable_gpu_data());
      if (local_decay) {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->gpu_data(),
            history_[param_id]->mutable_gpu_data());
      }
      // copy
      caffe_gpu_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void MaxIterTerminationCriterion<Dtype >::NotifyIteration(int iter) {
  this->criterion_met_ = iter >= max_iter_;
}
  
template <typename Dtype>
void TestAccuracyTerminationCriterion<Dtype >::NotifyTestAccuracy(Dtype test_accuracy) {
  if (test_accuracy > best_accuracy_) {
    //reset countdown
    count_down_ = test_accuracy_stop_countdown_;
    this->criterion_met_ = false;
    best_accuracy_ = test_accuracy;
  } else {
    --count_down_;
    if (count_down_ <= 0) {
      this->criterion_met_ = true;
    } else {
      this->criterion_met_ = false;
    }
  }
}


template <typename Dtype>
ExternalTerminationCriterion<Dtype >::ExternalTerminationCriterion(const std::string& cmd,
    int run_every_x_iterations)
 : cmd_(cmd),
   run_every_x_iterations_(run_every_x_iterations),
   learning_curve_file_("learning_curve.txt") {
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::NotifyTestAccuracy(Dtype test_accuracy) {
  learning_curve_file_ << test_accuracy << std::endl;
  learning_curve_file_.flush();
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::NotifyIteration(int iter) {
  if (iter % run_every_x_iterations_ == 0) {
    run();
  }
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::run() {
  int ret = system(cmd_.c_str());
  if (ret) {
    this->criterion_met_ = true;
  } else {
    this->criterion_met_ = false;
  }
}


INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(MaxIterTerminationCriterion);
INSTANTIATE_CLASS(TestAccuracyTerminationCriterion);
INSTANTIATE_CLASS(ExternalTerminationCriterion);


}  // namespace caffe
