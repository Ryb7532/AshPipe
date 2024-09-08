// Copyright (c) 2024 Ryb7532.
// Licensed under the MIT license.


#include <iostream>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

#define default_tuple make_tuple(-1e9, make_tuple(-1,-1), -1)
typedef tuple<double, tuple<int,int>, int> part_info;


vector<vector<vector<vector<part_info>>>> compute_partitioning(
    vector<vector<double>> compute_times,
    vector<double> activation_sizes,
    vector<double> parameter_sizes,
    vector<double> output_activation_sizes,
    vector<vector<int>> all_predecessor,
    vector<int> all_num_machines,
    vector<int> num_machines_list,
    vector<double> bandwidths,
    vector<double> p2p_bandwidths,
    vector<bool> connects,
    double memory_size,
    bool use_memory_constraint,
    bool straight_pipeline,
    bool use_fewer_machines,
    bool use_recomp,
    double activation_compression_ratio)
{
  vector<set<int>> all_predecessor_ids(
      all_predecessor.size());
  for (int i=0; i < all_predecessor.size(); i++) {
    all_predecessor_ids[i] = set<int>(all_predecessor[i].begin(), all_predecessor[i].end());
  }
  vector<vector<vector<vector<part_info>>>> all_As;
  int all_machines = num_machines_list.back();
  double total_time = compute_times.back()[0];
  double standard_time = total_time / all_machines;

  for (int counter=0; counter < all_num_machines.size(); counter++) {
    int num_machines = all_num_machines[counter];
    double bandwidth = bandwidths[counter], p2p_bandwidth = p2p_bandwidths[counter];
    bool fully_connect = connects[counter];

    printf("Solving optimization problem with %d machines with inter-machine bandwidth of allreduce: %.2f, p2p: %.2f GB/s\n", num_machines, bandwidth / 1e9, p2p_bandwidth / 1e9);

    int num_machines_in_machine = num_machines_list[counter];
    bool final_level = (counter == all_num_machines.size()-1);
    vector<vector<vector<part_info>>> A(
      compute_times.size(),
      vector<vector<part_info>>(
        compute_times.size(),
        vector<part_info>(num_machines, default_tuple)
      )
    );

    // m+1 data parallel for layer i-j
    for (int i=0; i < compute_times.size(); i++) {
      double start_time = 0.0, start_activation = 0.0, start_parameter = 0.0, start_time_recomp = 0.0, input_activation_size = 0.0;
      if (i > 0) {
        start_time = compute_times[i-1][0];
        start_activation = activation_sizes[i-1];
        start_parameter = parameter_sizes[i-1];
        start_time_recomp = compute_times[i-1][1];
        input_activation_size = output_activation_sizes[i-1];
      }
      for (int j=i; j < compute_times.size(); j++) {
        double cum_compute_time = compute_times[j][0];
        double cum_activation_size = activation_sizes[j];
        double cum_parameter_size = parameter_sizes[j];
        double cum_compute_time_recomp = compute_times[j][1];
        int num_versions = round((total_time - cum_compute_time) / standard_time) + 1;
        cum_compute_time -= start_time;
        cum_activation_size -= start_activation;
        cum_parameter_size -= start_parameter;
        cum_compute_time_recomp -= start_time_recomp;

        double stashed_data_size = num_versions * (cum_activation_size + cum_parameter_size) + cum_parameter_size;
        double compute_time = (counter == 0) ? cum_compute_time : get<0>(all_As[counter-1][i][j][all_num_machines[counter-1]-1]);
        bool recomp = false;
        if (counter == 0 && use_memory_constraint &&
            stashed_data_size > memory_size) {
          stashed_data_size = (num_versions+1) * cum_parameter_size +
            cum_activation_size;
          if (use_recomp && stashed_data_size <= memory_size) {
            recomp = true;
          } else
            compute_time = -1e9;
        }

        double input_transfer_time = input_activation_size / p2p_bandwidth, output_transfer_time = 0.0;
        if (j < output_activation_sizes.size()-1)
          output_transfer_time = output_activation_sizes[j] / p2p_bandwidth;
        if (activation_compression_ratio > 0) {
          input_transfer_time /= activation_compression_ratio;
          output_transfer_time /= activation_compression_ratio;
        } else {
          input_transfer_time = 0.0;
          output_transfer_time = 0.0;
        }
        if (recomp) {
          output_transfer_time = max(output_transfer_time, cum_compute_time_recomp);
        }

        if (compute_time >= 0.0) {
          A[i][j][0] = make_tuple(
            compute_time+input_transfer_time+output_transfer_time,
            make_tuple(-1,-1), 1);
        }

        int max_m = straight_pipeline ? 1 : num_machines;
        for (int m=1; m < max_m; m++) {
          int total_m = (m+1) * num_machines_in_machine;
          stashed_data_size = num_versions *
            (cum_activation_size / total_m + cum_parameter_size) + cum_parameter_size;
          recomp = false;
          if (use_memory_constraint && stashed_data_size > memory_size) {
            stashed_data_size = (num_versions+1) * cum_parameter_size +
              cum_activation_size / total_m;
            if (use_recomp && stashed_data_size <= memory_size)
              recomp = true;
            else
              continue;
          }
          double data_parallel_communication_time = (2 * cum_parameter_size) / bandwidth;
          if (!fully_connect)
            data_parallel_communication_time *= total_m - 1;

          input_transfer_time = input_activation_size / p2p_bandwidth;
          output_transfer_time = 0.0;
          if (j < output_activation_sizes.size()-1)
            output_transfer_time = output_activation_sizes[j] / p2p_bandwidth;
          if (activation_compression_ratio > 0) {
            input_transfer_time /= activation_compression_ratio;
            output_transfer_time /= activation_compression_ratio;
          } else {
            input_transfer_time = 0.0;
            output_transfer_time = 0.0;
          }
          if (fully_connect) {
            input_transfer_time /= m+1;
            output_transfer_time /= m+1;
          }
          if (recomp) {
            output_transfer_time = max(output_transfer_time, cum_compute_time_recomp / total_m);
          }

          if (cum_compute_time >= 0.0) {
            A[i][j][m] = make_tuple(
              (cum_compute_time+data_parallel_communication_time) / total_m +
              input_transfer_time + output_transfer_time,
              make_tuple(-1,-1), (m+1));
          }
        }
      }
    }

    int max_i = final_level ? 1 : compute_times.size()+1;
    for (int i=0; i<max_i; i++) {
      for (int m=1; m<num_machines; m++) {
        for (int j=i+1; j<compute_times.size(); j++) {
          double min_pipeline_time = get<0>(A[i][j][m]);
          auto optimal_split = get<1>(A[i][j][m]);
          int optimal_num_machines = get<2>(A[i][j][m]);
          double few_comp_time = get<0>(A[i][j][m-1]);
          if (use_fewer_machines && m > 0 &&
            (min_pipeline_time < 0.0 ||
            (few_comp_time >= 0.0 && few_comp_time < min_pipeline_time))) {
            min_pipeline_time = few_comp_time;
            optimal_split = get<1>(A[i][j][m-1]);
            optimal_num_machines = get<2>(A[i][j][m-1]);
          }
          for (int k: all_predecessor_ids[j]) {
            if (i > 0 && all_predecessor_ids[i-1].find(k) != all_predecessor_ids[i-1].end())
              continue;
            int max_m_prime = straight_pipeline ? 2 : m + 1;

            for (int m_prime=1; m_prime < max_m_prime; m_prime++) {
              if (get<0>(A[i][k][m-m_prime]) < 0.0 || get<0>(A[k+1][j][m_prime-1]) < 0.0)
                continue;
              double pipeline_time = max(get<0>(A[i][k][m-m_prime]), get<0>(A[k+1][j][m_prime-1]));
              if (min_pipeline_time < 0.0 || min_pipeline_time > pipeline_time) {
                optimal_split = make_tuple(k, m-m_prime);
                optimal_num_machines = m_prime;
                min_pipeline_time = pipeline_time;
              }
            }
          }
          A[i][j][m] = make_tuple(min_pipeline_time, optimal_split, optimal_num_machines);
        }
      }
    }
    all_As.push_back(A);
  }
  return all_As;
}

PYBIND11_MODULE(cppoptimizer, m) {
  m.doc() = "optimizer's plugin";
  m.def("compute_partitioning", &compute_partitioning, "A function that computes model partitions in C++");
}
