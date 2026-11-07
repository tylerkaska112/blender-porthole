/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include <map>
#  include <vector>

#  include "device/kernel.h"
#  include "device/memory.h"
#  include "device/queue.h"

#  include "device/metal_wm/device_impl.h"
#  include "device/metal_wm/wm.h"

#  define ANCILLARY_SLOT_COUNT 11

CCL_NAMESPACE_BEGIN

class MetalDeviceQueue : public DeviceQueue {
 public:
  MetalDeviceQueue(MetalDevice *device);
  ~MetalDeviceQueue() override;

  int num_concurrent_states(const size_t state_size) const override;
  int num_concurrent_busy_states(const size_t state_size) const override;
  int num_sort_partitions(int max_num_paths, uint max_scene_shaders) const override;

  bool supports_local_atomic_sort() const override;

  void init_execution() override;
  void load_image_info() override;

  bool enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) override;

  bool synchronize() override;

  void zero_to_device(device_memory &mem) override;
  void copy_to_device(device_memory &mem) override;
  void copy_from_device(device_memory &mem) override;
  void *copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage) override;

 protected:
  obj_handle_t get_compute_encoder(DeviceKernel kernel);
  obj_handle_t get_blit_encoder();
  void close_compute_encoder();
  void close_blit_encoder();
  void new_command_buffer();

  MetalDevice *metal_device_;
  Stats &stats_;

  obj_handle_t mtlCommandQueue_ = 0;
  obj_handle_t mtlCommandBuffer_ = 0;
  obj_handle_t mtlComputeEncoder_ = 0;
  obj_handle_t mtlBlitEncoder_ = 0;
  bool compute_encoder_concurrent_ = false;

  /* Diagnostics on stderr (Porthole logs it): what the queue is doing, and a summary of the pixels that come back. */
  void report_activity(const char *kernel_name);
  void log_readback(device_memory &mem);
  std::vector<device_memory *> pending_readbacks_;
  uint64_t n_enqueue_ = 0;
  uint64_t n_sync_ = 0;
  double last_report_ = 0.0;
  double report_started_ = 0.0;
  double encode_seconds_ = 0.0;
  double wait_seconds_ = 0.0;
  std::map<string, int> readback_counts_;
  void log_state_summary();
  string describe_address(uint64_t address);
  void probe_counter_write();
  void layout_check();
  bool layout_done_ = false;
  bool probe_done_ = false;
  DeviceKernel last_kernel_ = DEVICE_KERNEL_NUM;
  int n_state_logs_ = 0;
  int dispatch_logs_[DEVICE_KERNEL_NUM] = {};
};

CCL_NAMESPACE_END

#endif /* WITH_METAL */
