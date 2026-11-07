/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include <atomic>
#  include <map>
#  include <mutex>
#  include <thread>
#  include <vector>

#  include "device/device.h"
#  include "device/memory.h"
#  include "device/metal_wm/device.h"
#  include "device/metal_wm/wm.h"

#  include "kernel/device/metal/globals.h"

#  include "util/map.h"
#  include "util/unique_ptr.h"

CCL_NAMESPACE_BEGIN

class DeviceQueue;

/* What the queue needs to launch one kernel. Built the first time a kernel is used. */
struct MetalPipeline {
  obj_handle_t function = 0;
  obj_handle_t pipeline = 0;
  int num_threads_per_block = 0;
  bool failed = false;
};

class MetalDevice : public Device {
 public:
  obj_handle_t mtlDevice = 0;
  obj_handle_t mtlLibrary = 0;
  obj_handle_t mtlComputeCommandQueue = 0;
  obj_handle_t mtlResidencySet = 0;
  bool residency_dirty = false;
  std::mutex residency_mutex;

  string source;
  string library_features_key;

  wm::Buffer launch_params_buffer;
  KernelParamsMetal *launch_params = nullptr;

  uint kernel_features = 0;
  int max_threads_per_threadgroup = 512;
  bool apple9 = false; /* M3 and later: Cycles tunes these for 64 threads */

  /* Software BVH only for now (no hardware ray tracing in this port yet). */
  bool use_metalrt = false;

  MetalPipeline pipelines[DEVICE_KERNEL_NUM];
  std::mutex pipeline_mutex;
  std::mutex heavy_compile_mutex;

  /* The pipelines are built up front on a few threads (the first sample would otherwise wait for all of them,
   * one after the other, and Blender would show nothing). is_ready() reports the progress. */
  std::vector<std::thread> precompile_threads;
  std::atomic<int> precompile_next{0};
  std::atomic<int> precompile_done{0};
  std::atomic<int> precompile_active{0};
  int precompile_total = 0;
  void start_precompile();
  string preprocess_source_for(uint features);
  void join_precompile();
  void wait_for_kernels(double max_seconds);
  void log_memory_breakdown(const char *when);
  bool drop_denoising = false;
  string precompile_error;
  std::atomic<int> precompile_failed_kernel{-1};
  bool try_variant(uint features, DeviceKernel kernel, string &why);
  void recover_after_failure(DeviceKernel kernel);
  bool build_pipeline(DeviceKernel kernel, MetalPipeline &out, string &error_text);

  struct MetalMem {
    device_memory *mem = nullptr;
    int pointer_index = -1;
    wm::Buffer buffer;
    uint64_t offset = 0;
    uint64_t size = 0;
    void *hostPtr = nullptr;
  };
  using MetalMemMap = map<device_memory *, unique_ptr<MetalMem>>;
  MetalMemMap metal_mem_map;
  std::recursive_mutex metal_mem_map_mutex;
  vector<obj_handle_t> delayed_free_list;

  device_vector<KernelImageInfo> image_info;
  wm::Buffer image_bindings;

  /* What each image id points at: a 2D texture (its GPU resource id goes into the bindings
   * buffer) or a linear buffer (1D images: its GPU address goes into the image info). */
  struct ImageRes {
    obj_handle_t handle = 0;
    uint64_t gpu_id = 0;
    uint64_t gpu_address = 0;
    bool is_buffer = false;
  };
  vector<ImageRes> image_res;
  struct TextureMem {
    obj_handle_t handle = 0;
    size_t size = 0;
  };
  map<device_memory *, TextureMem> texture_map;

  bool has_error = false;

  MetalDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
  ~MetalDevice() override;

  void set_error(const string &error) override;
  bool is_ready(string &status) const override;
  void cancel() override {}

  BVHLayoutMask get_bvh_layout_mask(uint /*kernel_features*/) const override
  {
    return BVH_LAYOUT_BVH2;
  }

  bool check_peer_access(Device * /*peer_device*/) override
  {
    return false;
  }

  bool load_kernels(const uint kernel_features) override;
  /* The pipeline of a kernel, built on first use (null and an error on failure). */
  const MetalPipeline *get_pipeline(DeviceKernel kernel, bool quiet = false);

  unique_ptr<DeviceQueue> gpu_queue_create() override;

  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;
  void mem_copy_from(device_memory &mem)
  {
    mem_copy_from(mem, -1, -1, -1, -1);
  }
  void mem_copy_from(
      device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) override;
  void mem_or_from_device(device_memory &mem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;
  device_ptr mem_alloc_sub_ptr(device_memory &mem, const size_t offset, size_t size) override;

  void const_copy_to(const char *name, void *host, const size_t size) override;

  void global_alloc(device_memory &mem);
  void global_free(device_memory &mem);
  void image_alloc(device_image &mem);
  void image_alloc_as_buffer(device_image &mem);
  void image_copy_to(device_image &mem);
  void image_free(device_image &mem);
  void image_res_grow(const uint image_info_id);
  bool is_texture(const KernelImageInfo &info) const
  {
    return info.height > 0;
  }

  bool has_unified_memory_any() const override
  {
    return true;
  }

  MetalMem *generic_alloc(device_memory &mem);
  void generic_free(device_memory &mem);
  void erase_allocation(device_memory &mem);

  void load_image_info();
  void flush_delayed_free_list();

  /* Memory the GPU may use: kept in one residency set that the queue holds. */
  void metal_mem_alloc(const wm::Buffer &buffer);
  void metal_mem_alloc(obj_handle_t handle, size_t size);
  void metal_mem_free(obj_handle_t &handle, size_t size);
  void prepare_residency();

  size_t recommended_working_set() const;
  /* What all live Metal devices of this process hold (the viewport and a final render are two devices). */
  size_t total_mem_used() const;
  /* Runs a tiny kernel through the same path Cycles uses (GPU addresses in the argument bytes, residency set) and logs whether the GPU read, wrote and counted correctly. */
  void self_test();

  string preprocess_source();
  bool compile_library();
};

CCL_NAMESPACE_END

#endif /* WITH_METAL */
