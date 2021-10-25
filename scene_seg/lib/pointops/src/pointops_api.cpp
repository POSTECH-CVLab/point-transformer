#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ballquery/ballquery_cuda_kernel.h"
#include "grouping/grouping_cuda_kernel.h"
#include "grouping_int/grouping_int_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "interpolation/interpolation_cuda_kernel.h"
#include "knnquery/knnquery_cuda_kernel.h"
#include "knnquery_heap/knnquery_heap_cuda_kernel.h"
#include "labelstat/labelstat_cuda_kernel.h"
#include "featuredistribute/featuredistribute_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ballquery_cuda", &ballquery_cuda_fast, "ballquery_cuda_fast");   // name in python, cpp function address, docs

    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    m.def("knnquery_heap_cuda", &knnquery_heap_cuda, "knnquery_heap_cuda");

    m.def("grouping_forward_cuda", &grouping_forward_cuda_fast, "grouping_forward_cuda_fast");
    m.def("grouping_backward_cuda", &grouping_backward_cuda, "grouping_backward_cuda");

    m.def("grouping_int_forward_cuda", &grouping_int_forward_cuda_fast, "grouping_int_forward_cuda_fast");

    m.def("gathering_forward_cuda", &gathering_forward_cuda, "gathering_forward_cuda");
    m.def("gathering_backward_cuda", &gathering_backward_cuda, "gathering_backward_cuda");
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");

    m.def("nearestneighbor_cuda", &nearestneighbor_cuda_fast, "nearestneighbor_cuda_fast");
    m.def("interpolation_forward_cuda", &interpolation_forward_cuda_fast, "interpolation_forward_cuda_fast");
    m.def("interpolation_backward_cuda", &interpolation_backward_cuda, "interpolation_backward_cuda");

    m.def("labelstat_idx_cuda", &labelstat_idx_cuda_fast, "labelstat_idx_cuda_fast");
    m.def("labelstat_ballrange_cuda", &labelstat_ballrange_cuda_fast, "labelstat_ballrange_cuda_fast");
    m.def("labelstat_and_ballquery_cuda", &labelstat_and_ballquery_cuda_fast, "labelstat_and_ballquery_cuda_fast");

    m.def("featuredistribute_cuda", &featuredistribute_cuda, "featuredistribute_cuda");
    m.def("featuregather_forward_cuda", &featuregather_forward_cuda, "featuregather_forward_cuda");
    m.def("featuregather_backward_cuda", &featuregather_backward_cuda, "featuregather_backward_cuda");
}
