/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "device/metal_wm/queue.h"

#  include <cmath>

#  include "util/half.h"
#  include "util/time.h"

#  include "util/debug.h"
#  include "util/log.h"
#  include "util/time.h"

CCL_NAMESPACE_BEGIN

MetalDeviceQueue::MetalDeviceQueue(MetalDevice *device)
    : DeviceQueue(device), metal_device_(device), stats_(device->stats)
{
  mtlCommandQueue_ = device->mtlComputeCommandQueue;
}

MetalDeviceQueue::~MetalDeviceQueue()
{
  /* An encoder that is released without endEncoding aborts the process (Metal assertion), also after an error. */
  close_compute_encoder();
  close_blit_encoder();
  synchronize();
}

int MetalDeviceQueue::num_concurrent_states(const size_t state_size) const
{
#  ifdef _WIN32
  /* Under Porthole (a phone): 524288 paths keep the viewport and a final render inside the memory limit. */
  size_t state_count = 524288;
#  else
  size_t state_count = 4194304;
#  endif
  if (auto *str = getenv("CYCLES_METAL_MAX_STATES")) {
    if (atol(str) > 0) {
      state_count = size_t(atol(str));
    }
  }

  const size_t max_recommended_working_set = metal_device_->recommended_working_set();

  /* Allow for a percentage of the recommended working set to be used (default 90%). */
  size_t percent = 90;
  if (auto *str = getenv("CYCLES_METAL_WORKING_SET_PERCENT")) {
    percent = atoi(str);
  }
  const size_t max_working_set = (max_recommended_working_set * percent) / 100;

  /* Calculate how many states fit in the memory that is left. */
  size_t max_safe_state_count = 0;
  const size_t mem_used = metal_device_->total_mem_used();
  if (mem_used < max_working_set) {
    const size_t headroom = max_working_set - mem_used;
    max_safe_state_count = headroom / state_size;
  }

  if (max_safe_state_count >= 65536) {
    if (max_safe_state_count < state_count) {
      LOG_INFO << "Reducing state count to fit within available RAM. " << state_count << " -> "
               << max_safe_state_count;
      state_count = max_safe_state_count;
    }
    else {
      size_t grow_percent = 50;
      if (auto *str = getenv("CYCLES_METAL_GROW_PERCENT")) {
        grow_percent = atoi(str);
      }
      max_safe_state_count = (max_safe_state_count * grow_percent) / 100;
      for (int i = 0; i < 2; i++) {
        if (max_safe_state_count > state_count * 2) {
          state_count *= 2;
        }
      }
    }
  }
  else {
    metal_device_->set_error("Out of memory - couldn't allocate integrator state");
    state_count = 0;
  }

  fprintf(stderr,
          "[cycles-wm] integrator states: %zu x %zu bytes, budget %zu MB, in use %zu MB\n",
          size_t(state_count),
          size_t(state_size),
          max_recommended_working_set >> 20,
          mem_used >> 20);
  return state_count;
}

int MetalDeviceQueue::num_concurrent_busy_states(const size_t state_size) const
{
  return num_concurrent_states(state_size) / 4;
}

int MetalDeviceQueue::num_sort_partitions(int max_num_paths, uint max_scene_shaders) const
{
  int sort_partition_elements = 65536;
  if (auto *str = getenv("CYCLES_METAL_SORT_PARTITION_ELEMENTS")) {
    sort_partition_elements = atoi(str);
  }

  /* Partitioning the active indices before sorting each partition by material helps the cache on
   * smaller scenes. */
  if (max_scene_shaders < 300 && sort_partition_elements > 0) {
    return max(max_num_paths / sort_partition_elements, 1);
  }
  return 1;
}

bool MetalDeviceQueue::supports_local_atomic_sort() const
{
  return DebugFlags().metal.use_local_atomic_sort;
}

void MetalDeviceQueue::init_execution()
{
  load_image_info();
  synchronize();
}

void MetalDeviceQueue::load_image_info()
{
  metal_device_->load_image_info();
}

void MetalDeviceQueue::new_command_buffer()
{
  if (!mtlCommandBuffer_) {
    mtlCommandBuffer_ = MTLCommandQueue_commandBuffer(mtlCommandQueue_);
  }
}

obj_handle_t MetalDeviceQueue::get_compute_encoder(DeviceKernel kernel)
{
  /* The kernels of the integrator may run concurrently (they synchronize through the queue counters). */
  const bool concurrent = int(kernel) < int(DEVICE_GPU_KERNEL_INTEGRATOR_NUM);

  if (mtlComputeEncoder_) {
    if (compute_encoder_concurrent_ == concurrent) {
      return mtlComputeEncoder_;
    }
    close_compute_encoder();
  }

  close_blit_encoder();
  new_command_buffer();

  mtlComputeEncoder_ = MTLCommandBuffer_computeCommandEncoder(mtlCommandBuffer_, concurrent);
  compute_encoder_concurrent_ = concurrent;
  return mtlComputeEncoder_;
}

obj_handle_t MetalDeviceQueue::get_blit_encoder()
{
  if (mtlBlitEncoder_) {
    return mtlBlitEncoder_;
  }

  close_compute_encoder();
  new_command_buffer();

  mtlBlitEncoder_ = MTLCommandBuffer_blitCommandEncoder(mtlCommandBuffer_);
  return mtlBlitEncoder_;
}

void MetalDeviceQueue::close_compute_encoder()
{
  if (mtlComputeEncoder_) {
    MTLCommandEncoder_endEncoding(mtlComputeEncoder_);
    mtlComputeEncoder_ = 0;
  }
}

void MetalDeviceQueue::close_blit_encoder()
{
  if (mtlBlitEncoder_) {
    MTLCommandEncoder_endEncoding(mtlBlitEncoder_);
    mtlBlitEncoder_ = 0;
  }
}

void MetalDeviceQueue::report_activity(const char *kernel_name)
{
  /* Once every 3 seconds at most: how many kernels went to the GPU and how often the queue waited for it. */
  const double now = time_dt();
  if (now - last_report_ >= 3.0) {
    last_report_ = now;
    fprintf(stderr,
            "[cycles-wm] queue: %llu kernel dispatches, %llu synchronizations so far, last kernel %s\n",
            (unsigned long long)n_enqueue_,
            (unsigned long long)n_sync_,
            kernel_name);
  }
}

void MetalDeviceQueue::log_readback(device_memory &mem)
{
  /* Summary of what the GPU wrote: all zeros would mean the kernels ran but produced nothing. */
  if (n_readbacks_logged_ >= 40 || !mem.host_pointer || mem.memory_size() == 0) {
    return;
  }
  size_t count = 0;
  const bool is_float = (mem.data_type == TYPE_FLOAT);
  const bool is_half = (mem.data_type == TYPE_HALF);
  if (is_float) {
    count = mem.memory_size() / sizeof(float);
  }
  else if (is_half) {
    count = mem.memory_size() / sizeof(half);
  }
  else {
    return;
  }
  const size_t limit = std::min<size_t>(count, 8u << 20);
  size_t nonzero = 0, nans = 0;
  double sum = 0.0, vmin = 1e30, vmax = -1e30;
  for (size_t i = 0; i < limit; i++) {
    const float v = is_float ? ((const float *)mem.host_pointer)[i] :
                               half_to_float(((const half *)mem.host_pointer)[i]);
    if (v != v) {
      nans++;
      continue;
    }
    if (v != 0.0f) {
      nonzero++;
    }
    sum += v;
    vmin = std::min<double>(vmin, v);
    vmax = std::max<double>(vmax, v);
  }
  n_readbacks_logged_++;
  fprintf(stderr,
          "[cycles-wm] readback \"%s\": %zu values checked, %zu non-zero, %zu NaN, min %g, max %g, mean %g\n",
          mem.log_name().c_str(),
          limit,
          nonzero,
          nans,
          limit ? vmin : 0.0,
          limit ? vmax : 0.0,
          limit ? sum / double(limit) : 0.0);
}

bool MetalDeviceQueue::enqueue(DeviceKernel kernel,
                               const int work_size,
                               const DeviceKernelArguments &args)
{
  if (metal_device_->have_error()) {
    return false;
  }

  debug_enqueue_begin(kernel, work_size);
  n_enqueue_++;
  report_activity(device_kernel_as_string(kernel));

  const MetalPipeline *pipeline = metal_device_->get_pipeline(kernel);
  if (!pipeline) {
    metal_device_->set_error(
        string_printf("Could not activate pipeline for %s\n", device_kernel_as_string(kernel)));
    return false;
  }

  const obj_handle_t encoder = get_compute_encoder(kernel);

  /* Collect the arguments that change per dispatch: pointers become the GPU address of the buffer they refer to. */
  uint8_t dynamic_args[512] = {0};
  size_t dynamic_bytes_written = 0;
  size_t max_size_in_bytes = 0;
  for (size_t i = 0; i < args.count; i++) {
    const size_t size_in_bytes = args.sizes[i];
    max_size_in_bytes = max(max_size_in_bytes, size_in_bytes);
    dynamic_bytes_written = round_up(dynamic_bytes_written, size_in_bytes);
    memcpy(dynamic_args + dynamic_bytes_written, args.values[i], size_in_bytes);
    if (args.types[i] == DeviceKernelArguments::POINTER) {
      uint64_t *pptr = (uint64_t *)(dynamic_args + dynamic_bytes_written);
      if (MetalDevice::MetalMem *mmem = (MetalDevice::MetalMem *)pptr[0]) {
        pptr[0] = mmem->buffer.gpu_address;
      }
    }
    dynamic_bytes_written += size_in_bytes;
  }
  /* Apple GPUs require the size of the argument data to be a multiple of its largest member. */
  dynamic_bytes_written = round_up(dynamic_bytes_written, max_size_in_bytes);
  assert(dynamic_bytes_written <= sizeof(dynamic_args));

  /* The image bindings (empty until textures are supported). */
  uint64_t ancillary_args[ANCILLARY_SLOT_COUNT] = {0};
  ancillary_args[0] = metal_device_->image_bindings.gpu_address;

  /* Threadgroup memory some of the path-array and sort kernels use. */
  const int num_threads_per_block = pipeline->num_threads_per_block;
  int shared_mem_bytes = 0;
  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY:
      shared_mem_bytes = (int)round_up((num_threads_per_block + 1) * sizeof(int), 16);
      break;

    case DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS:
    case DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS: {
      /* Shared memory for the sort buckets. */
      const int key_count = metal_device_->launch_params->data.max_shaders;
      shared_mem_bytes = (int)round_up(key_count * sizeof(int), 16);
      break;
    }

    default:
      break;
  }

  /* Without a residency set every buffer has to be named for the encoder. */
  vector<wmtcmd_compute_useresource> uses;
  if (!metal_device_->mtlResidencySet) {
    std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
    uses.reserve(metal_device_->metal_mem_map.size() + 2);
    for (auto &it : metal_device_->metal_mem_map) {
      if (!it.second->buffer.handle) {
        continue;
      }
      wmtcmd_compute_useresource use = {};
      use.type = WMTComputeCommandUseResource;
      use.resource = it.second->buffer.handle;
      use.usage = WMTResourceUsage(WMTResourceUsageRead | WMTResourceUsageWrite);
      uses.push_back(use);
    }
    for (const obj_handle_t extra :
         {metal_device_->launch_params_buffer.handle, metal_device_->image_bindings.handle})
    {
      wmtcmd_compute_useresource use = {};
      use.type = WMTComputeCommandUseResource;
      use.resource = extra;
      use.usage = WMTResourceUsage(WMTResourceUsageRead | WMTResourceUsageWrite);
      uses.push_back(use);
    }
  }

  wmtcmd_compute_setpso set_pso = {};
  set_pso.type = WMTComputeCommandSetPSO;
  set_pso.pso = pipeline->pipeline;
  set_pso.threadgroup_size = {(uint64_t)num_threads_per_block, 1, 1};

  wmtcmd_compute_setbytes set_dynamic = {};
  set_dynamic.type = WMTComputeCommandSetBytes;
  set_dynamic.bytes.ptr = dynamic_args;
  set_dynamic.length = dynamic_bytes_written;
  set_dynamic.index = 0;

  wmtcmd_compute_setbuffer set_launch = {};
  set_launch.type = WMTComputeCommandSetBuffer;
  set_launch.buffer = metal_device_->launch_params_buffer.handle;
  set_launch.offset = 0;
  set_launch.index = 1;

  wmtcmd_compute_setbytes set_ancillary = {};
  set_ancillary.type = WMTComputeCommandSetBytes;
  set_ancillary.bytes.ptr = ancillary_args;
  set_ancillary.length = sizeof(ancillary_args);
  set_ancillary.index = 2;

  wmtcmd_compute_setthreadgroupmemory set_shared = {};
  set_shared.type = WMTComputeCommandSetThreadgroupMemoryLength;
  set_shared.length = shared_mem_bytes;
  set_shared.index = 0;

  wmtcmd_compute_dispatch dispatch = {};
  dispatch.type = WMTComputeCommandDispatchThreads;
  dispatch.size = {(uint64_t)work_size, 1, 1};

  /* One chain: resources, state, arguments, threadgroup memory, dispatch. */
  wmtcmd_base *head = nullptr;
  wmtcmd_base **tail = &head;
  auto link = [&](void *cmd) {
    wmtcmd_base *base = (wmtcmd_base *)cmd;
    base->next.ptr = nullptr;
    *tail = base;
    tail = (wmtcmd_base **)&base->next.ptr;
  };
  for (wmtcmd_compute_useresource &use : uses) {
    link(&use);
  }
  link(&set_pso);
  link(&set_dynamic);
  link(&set_launch);
  link(&set_ancillary);
  if (shared_mem_bytes) {
    assert(shared_mem_bytes <= 32 * 1024);
    link(&set_shared);
  }
  link(&dispatch);

  MTLComputeCommandEncoder_encodeCommands(encoder, head);

  metal_device_->prepare_residency();

  debug_enqueue_end();

  return !(metal_device_->have_error());
}

bool MetalDeviceQueue::synchronize()
{
  /* Always end the open encoders, even when the device is in error: dropping one open aborts the process. */
  close_compute_encoder();
  close_blit_encoder();

  if (metal_device_->have_error()) {
    return false;
  }

  if (mtlCommandBuffer_) {
    MTLCommandBuffer_commit(mtlCommandBuffer_);
    MTLCommandBuffer_waitUntilCompleted(mtlCommandBuffer_);

    const WMTCommandBufferStatus status = MTLCommandBuffer_status(mtlCommandBuffer_);
    mtlCommandBuffer_ = 0;
    if (status != WMTCommandBufferStatusCompleted) {
      fprintf(stderr, "[cycles-wm] command buffer finished with status %d\n", int(status));
      metal_device_->set_error(
          string_printf("Command buffer not completed. status = %d.", int(status)));
    }

    /* The GPU is done: what was freed since the last synchronize can go. */
    metal_device_->flush_delayed_free_list();
  }

  n_sync_++;
  for (device_memory *mem : pending_readbacks_) {
    log_readback(*mem);
  }
  pending_readbacks_.clear();

  debug_synchronize();

  return !(metal_device_->have_error());
}

void MetalDeviceQueue::zero_to_device(device_memory &mem)
{
  if (metal_device_->have_error()) {
    return;
  }

  assert(mem.type != MEM_IMAGE_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    metal_device_->mem_alloc(mem);
  }

  /* Zero memory on device. */
  assert(mem.device_pointer != 0);

  std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
  MetalDevice::MetalMem &mmem = *metal_device_->metal_mem_map.at(&mem);
  if (mmem.buffer.handle) {
    /* In the queue's order, so kernels that are still waiting to run are not overtaken. */
    const obj_handle_t blit = get_blit_encoder();
    wmtcmd_blit_fillbuffer fill = {};
    fill.type = WMTBlitCommandFillBuffer;
    fill.buffer = mmem.buffer.handle;
    fill.offset = mmem.offset;
    fill.length = mmem.size;
    fill.value = 0;
    fill.next.ptr = nullptr;
    MTLBlitCommandEncoder_encodeCommands(blit, (const wmtcmd_base *)&fill);
  }
  else {
    metal_device_->mem_zero(mem);
  }
}

void MetalDeviceQueue::copy_to_device(device_memory &mem)
{
  if (metal_device_->have_error()) {
    return;
  }

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. The host side is the buffer itself (unified memory), so there is nothing to copy. */
  if (mem.device_pointer == 0) {
    metal_device_->mem_alloc(mem);
  }
}

void MetalDeviceQueue::copy_from_device(device_memory &mem)
{
  /* Unified memory: the host reads the buffer in place after the queue was synchronized. */
  if (n_readbacks_logged_ < 40) {
    pending_readbacks_.push_back(&mem);
  }
}

void *MetalDeviceQueue::copy_from_device_synchronized(device_memory &mem,
                                                      vector<uint8_t> & /*storage*/)
{
  if (mem.memory_size() == 0) {
    return nullptr;
  }

  synchronize();

  const device_ptr d_ptr = mem.device->mem_device_ptr(mem, metal_device_);
  return (d_ptr) ? reinterpret_cast<MetalDevice::MetalMem *>(d_ptr)->hostPtr : nullptr;
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
