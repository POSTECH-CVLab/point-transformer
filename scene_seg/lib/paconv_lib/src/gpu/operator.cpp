#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("assign_score_withk_forward_cuda", &assign_score_withk_forward_kernel_wrapper, "Assign score kernel forward (GPU), save memory version");
  m.def("assign_score_withk_backward_cuda", &assign_score_withk_backward_kernel_wrapper, "Assign score kernel backward (GPU), save memory version");
}
