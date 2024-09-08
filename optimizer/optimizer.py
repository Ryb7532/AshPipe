# Copyright (c) 2024 Ryb7532.
# Licensed under the MIT license.


from __future__ import print_function

import argparse
from collections import OrderedDict
import json
import os

import sys

sys.path.append("..")
import graph

from cppoptimizer import compute_partitioning


def analyze_partitioning(A, states, start, end, allreduce_bandwidth, p2p_bandwidth,
                         fully_connect, num_machines, num_machines_in_machine,
                         activation_compression_ratio, print_configuration, verbose):
    metadata = A[start][end-1][num_machines-1]
    next_split = None if metadata[1][0] < 0 else metadata[1]
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    prev_split = end
    while next_split is not None:
        num_machines_used = None if metadata[2] < 0 else metadata[2]
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0]+1, next_split[0]+2))
            print("Split before antichain %s..." % (states[next_split[0]+1].antichain))
        compute_time = states[prev_split-1].compute_time - states[next_split[0]].compute_time
        total_num_machines_used = 1

        if num_machines_used != 1:
            total_num_machines_used = num_machines_used * num_machines_in_machine
            splits.append((prev_split, total_num_machines_used))
            parameter_size = states[prev_split-1].parameter_size - \
                states[next_split[0]].parameter_size

            dp_communication_time = (2 * parameter_size) \
                / (allreduce_bandwidth * total_num_machines_used)
            if not fully_connect:
                dp_communication_time *= total_num_machines_used - 1
        else:
            splits.append((prev_split, num_machines_used))
            dp_communication_time = 0.0
        compute_time /= total_num_machines_used
        pp_communication_time_input = \
            states[next_split[0]].output_activation_size / p2p_bandwidth
        pp_communication_time_output = \
            states[prev_split-1].output_activation_size / p2p_bandwidth
        if activation_compression_ratio is not None:
            pp_communication_time_input /= activation_compression_ratio
            pp_communication_time_output /= activation_compression_ratio
        if activation_compression_ratio is None:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0
        if fully_connect:
            pp_communication_time_input /= num_machines_used
            pp_communication_time_output /= num_machines_used

        if verbose:
            print(("Compute time = %f, Data-parallel communication time = %f, "
                   "Pipeline-parallel communication time = %f...") % (
                compute_time, dp_communication_time,
                (pp_communication_time_input + pp_communication_time_output)))
        prev_split = next_split[0]+1
        metadata = A[start][next_split[0]][next_split[1]]
        next_split = None if metadata[1][0] < 0 else metadata[1]
        replication_factors.append(num_machines_used)
        remaining_machines_left -= num_machines_used
    if verbose:
        print("-------------------------------------")
        print("Number of machines used: %d..." % metadata[2])

    num_machines_used = None if metadata[2] < 0 else metadata[2]
    total_num_machines_used = 1
    if num_machines_used > 1:
        total_num_machines_used = num_machines_used * num_machines_in_machine
        splits.append((prev_split, total_num_machines_used))
    else:
        splits.append((prev_split, num_machines_used))
    replication_factors.append(num_machines_used)
    remaining_machines_left -= num_machines_used
    if start == 0:
        compute_time = states[prev_split-1].compute_time
        parameter_size = states[prev_split-1].parameter_size
    else:
        compute_time = states[prev_split-1].compute_time - states[start-1].compute_time
        parameter_size = states[prev_split-1].parameter_size - states[start-1].parameter_size
    dp_communication_time = ((2 * parameter_size) /
                             (allreduce_bandwidth * total_num_machines_used))
    if not fully_connect:
        dp_communication_time *= total_num_machines_used - 1
    compute_time /= total_num_machines_used

    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." %
              (compute_time, dp_communication_time))
        print("-------------------------------------")
    if print_configuration:
        print("Number of machines in budget not used: %d..." %
              remaining_machines_left)
        print()
        print("(Split start, split end) / compute time taken per stage "
              "/ replication factor per stage:")
    prev_split = start
    splits.reverse()
    replication_factors.reverse()
    for i in range(len(splits)):
        time = 0.0
        if prev_split > 0:
            time = states[splits[i][0]-1].compute_time - states[prev_split-1].compute_time
        else:
            time = states[splits[i][0]-1].compute_time
        if print_configuration:
            print((prev_split+1, splits[i][0]), time, replication_factors[i])
        prev_split = splits[i][0]
    if print_configuration:
        print()
    return splits

def main(all_num_machines, profile_filename, network_conf, memory_size,
         use_memory_constraint, straight_pipeline, use_fewer_machines,
         activation_compression_ratio, minibatch_ratio, use_recompute,
         compress_graph, output_directory, print_configuration=True, verbose=False):
    gr = graph.Graph.from_str(open(profile_filename, 'r').read())

    if compress_graph:
        gr = gr.compress_branches()
    sources = gr.sources()
    nodes_to_remove = OrderedDict()
    host_to_device_time = 0.0
    data_load_time = 0.0
    for source in sources:
        if source.node_desc == 'Input0':
            host_to_device_time = source.forward_compute_time / 1000.0 * minibatch_ratio
            data_load_time = source.backward_compute_time / 1000.0 * minibatch_ratio
        if source.node_desc.startswith("Input"):
            source.forward_compute_time = 0.0
            source.backward_compute_time = 0.0
            source.activation_size = 0.0
            source.parameter_size = 0.0
            nodes_to_remove[source] = []
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)
            gr.remove_node(source)

    sinks = gr.sinks()
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            gr.remove_node(sink)

    antichain_gr = gr.antichain_dag()
    states = antichain_gr.topological_sort()
    if verbose:
        print("Total number of states: %d" % len(states))
        print("Minibatch ratio: %f" % minibatch_ratio)
        print("Memory size: %f" % memory_size)

    states_indices = {}
    for i in range(len(states)):
        states_indices[states[i]] = i
    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size

    for i in range(len(states)):
        antichain = states[i].antichain
        all_predecessors = gr.all_predecessors(antichain)
        states[i].compute_time = 0.0
        states[i].recompute_time = 0.0
        states[i].activation_size = 0.0
        states[i].parameter_size = 0.0
        for predecessor in all_predecessors:
            states[i].compute_time += (predecessor.forward_compute_time +
                predecessor.backward_compute_time) / 1000.0 * minibatch_ratio
            states[i].recompute_time += predecessor.forward_compute_time / \
                1000.0 * minibatch_ratio
            if not ("inplace" in predecessor.node_desc or "Add" in predecessor.node_desc):
                states[i].activation_size += predecessor.activation_size * minibatch_ratio
            states[i].parameter_size += predecessor.parameter_size

    output_activation_sizes = [state.output_activation_size for state in states]
    all_predecessor_ids = [[states_indices[predecessor] for predecessor in
                            antichain_gr.predecessors(states[i].node_id)]
                           for i in range(len(states))]

    compute_times = []
    activation_sizes = []
    parameter_sizes = []
    for i in range(len(states)):
        compute_times.append([
            max(states[i].compute_time + host_to_device_time, data_load_time),
            states[i].recompute_time])
        activation_sizes.append(states[i].activation_size)
        parameter_sizes.append(states[i].parameter_size)

    total_num_machines = 1
    num_machines_list = [1]
    for n in all_num_machines:
        total_num_machines *= n
        num_machines_list.append(total_num_machines)

    allreduce_bandwidths = network_conf["allreduce_bandwidths"]
    p2p_bandwidths = network_conf["p2p_bandwidths"]
    connects = network_conf["fully_connect"]

    all_As = compute_partitioning(
        compute_times, activation_sizes, parameter_sizes,
        output_activation_sizes, all_predecessor_ids,
        all_num_machines, num_machines_list,
        allreduce_bandwidths, p2p_bandwidths, connects,
        memory_size, use_memory_constraint, straight_pipeline,
        use_fewer_machines, use_recompute, activation_compression_ratio
    )

    splits = [(0, len(states), 1)]
    i = len(all_As) - 1
    while i >= 0:
        print("======================================")
        print("Level %d" % (i+1))
        print("======================================")
        new_splits = []
        stage_id = 0
        for (start, end, n) in splits:
            if n == 1:
                try:
                    partial_splits = \
                        analyze_partitioning(all_As[i], states, start, end,
                                            allreduce_bandwidths[i], p2p_bandwidths[i],
                                            connects[i], all_num_machines[i],
                                            num_machines_list[i],
                                            activation_compression_ratio,
                                            print_configuration, verbose)
                except:
                    print("RuntimeError: Expected out of memory. Please lower the minibatch size.")
                    raise
            else:
                partial_splits = [(end, n)]
            start_point = start
            for (split, n) in partial_splits:
                new_splits.append((start_point, split, n))
                if i == 0:
                    predecessors = gr.all_predecessors(states[split-1].antichain)
                    for predecessor in predecessors:
                        if predecessor.stage_id is None:
                            predecessor.set_stage_id(stage_id)
                start_point = split
                stage_id += 1
        splits = new_splits
        i -= 1

    print("stage id / (stage start, stage end) / num replicas / recompute")
    stage_conf = ""
    for i, (start, end, n) in enumerate(new_splits):
        if start > 0:
            parameter_size = states[end-1].parameter_size - states[start].parameter_size
            activation_size = states[end-1].activation_size - states[start].activation_size
        else:
            parameter_size = states[end-1].parameter_size
            activation_size = states[end-1].activation_size
        recomp = (stage_id - i) * (parameter_size + activation_size / n) + \
            parameter_size > memory_size
        stage_conf += f"{i}:{n},"
        print("stage {}: ({}, {}) {} {}".\
            format(i, start+1, end, n, recomp))
    print("Total number of stages: %d" % stage_id)
    print("stage_to_num_ranks_map: {}".format(stage_conf[0:-1]))

    for source in nodes_to_remove:
        for out_node in nodes_to_remove[source]:
            source.stage_id = 0
            gr.add_edge(source, out_node)

    if output_directory is not None:
        gr.to_dot(os.path.join(output_directory, "gpus=%d" % total_num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpus=%d.txt" % total_num_machines), 'w') as f:
            f.write(gr_str)

    total_time = states[-1].compute_time
    total_parameter_size = states[-1].parameter_size
    num_machines = all_num_machines[-1]
    allreduce_bandwidth = allreduce_bandwidths[-1]
    data_parallel_communication_time = (
        (2 * total_parameter_size) / allreduce_bandwidth)
    if not connects[-1]:
        data_parallel_communication_time *= total_num_machines - 1
    data_parallel_total_time = sum(
        [total_time, data_parallel_communication_time]) / total_num_machines
    pipeline_parallel_total_time = all_As[-1][0][len(states)-1][num_machines-1][0]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):",
              total_time / pipeline_parallel_total_time)
        dp_str = ",".join([str(elem) for elem in all_num_machines])
        print(("[Note that single-machine and (%s)-machine DP might not fit "
               "given memory constraints]") % dp_str)
        print("Throughput increase of (%s)-machine DP compared to single "
              "machine:" % dp_str, total_time / data_parallel_total_time)
        print("Throughput increase (compared to (%s)-machine DP):" % dp_str,
              data_parallel_total_time / pipeline_parallel_total_time)
    return pipeline_parallel_total_time, data_parallel_total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run AshPipe's optimizer")
    )
    parser.add_argument('-n', "--all_num_machines", nargs='+', type=int,
                        help="Number of machines available")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Profile filename")
    parser.add_argument('-b', "--network_bandwidths_filename", default=None, type=str,
                        help="Path to file including network config")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")
    parser.add_argument('-o', "--output_directory", default=None, type=str,
                        help="Output directory to dump processed graph")
    parser.add_argument("--use_memory_constraint", action='store_true',
                        help="Enforce memory constraint per machine")
    parser.add_argument("--straight_pipeline", action='store_true',
                        help="No replication across stages")
    parser.add_argument("--use_fewer_machines", action='store_true',
                        help="Use fewer machines, if possible")
    parser.add_argument("--activation_compression_ratio", default=None, type=float,
                        help="Compression ratio for activations")
    parser.add_argument("--minibatch_ratio", default=1.0, type=float,
                        help="ratio of minibatch size at runtime to at profiling")
    parser.add_argument("--use_recompute", action='store_true',
                        help="Use recompute technique, if necessary")
    parser.add_argument("--compress_graph", action='store_true',
                        help="Compress the model graph to reduce computation")

    args = parser.parse_args()
    args = vars(args)

    all_num_machines = args["all_num_machines"]
    profile_filename = args["profile_filename"]
    network_filename = args["network_bandwidths_filename"]
    network_conf = {
        "allreduce_bandwidths": [1000000000],
        "p2p_bandwidths": [1000000000],
        "fully_connect": [False]
    }
    if network_filename is not None:
        json_bw_file = json.load(open(network_filename, 'r'))
        network_conf["allreduce_bandwidths"] = json_bw_file.get("allreduce_bandwidths", [1000000000])
        network_conf["p2p_bandwidths"] = json_bw_file.get("p2p_bandwidths", [1000000000])
        network_conf["fully_connect"] = json_bw_file.get("fully_connect", [False])
    assert(len(all_num_machines) == len(network_conf["allreduce_bandwidths"]) and len(all_num_machines) == len(network_conf["p2p_bandwidths"]) and len(all_num_machines) == len(network_conf["fully_connect"]))
    memory_size = args["memory_size"]
    output_directory = args["output_directory"]
    use_memory_constraint = args["use_memory_constraint"]
    straight_pipeline = args["straight_pipeline"]
    use_fewer_machines = args["use_fewer_machines"]
    activation_compression_ratio = args["activation_compression_ratio"]
    minibatch_ratio = args["minibatch_ratio"]
    use_recompute = args["use_recompute"]
    compress_graph = args["compress_graph"]

    main(all_num_machines, profile_filename, network_conf, memory_size,
         use_memory_constraint, straight_pipeline, use_fewer_machines,
         activation_compression_ratio, minibatch_ratio, use_recompute,
         compress_graph, output_directory, verbose=True)
