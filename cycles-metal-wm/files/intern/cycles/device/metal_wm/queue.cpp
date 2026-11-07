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
  /* Under Porthole (a phone): 1048576 paths (365 MB of state) cover a whole viewport in one pass, which halves the round trips to the GPU
   * that the host waits for, and still sit well inside the memory limit. */
  size_t state_count = 1048576;
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
#  ifndef _WIN32
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
#  endif
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
  bool concurrent = int(kernel) < int(DEVICE_GPU_KERNEL_INTEGRATOR_NUM);
#  ifdef _WIN32
  /* Under Porthole the dispatches run one after the other: CYCLES_METAL_CONCURRENT=1 brings the overlap back. */
  const char *conc_env = getenv("CYCLES_METAL_CONCURRENT");
  if (!(conc_env && atoi(conc_env) != 0)) {
    concurrent = false;
  }
#  endif

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
            "[cycles-wm] queue: %llu kernel dispatches, %llu synchronizations so far, last kernel %s; since the last line: %.0f ms encoding, %.0f ms waiting for the GPU, %.1f s elapsed\n",
            (unsigned long long)n_enqueue_,
            (unsigned long long)n_sync_,
            kernel_name,
            encode_seconds_ * 1000.0,
            wait_seconds_ * 1000.0,
            report_started_ > 0 ? now - report_started_ : 0.0);
    encode_seconds_ = 0;
    wait_seconds_ = 0;
    report_started_ = now;
  }
}

void MetalDeviceQueue::log_readback(device_memory &mem)
{
  /* Summary of what the GPU wrote: all zeros would mean the kernels ran but produced nothing. */
  /* At most 12 summaries per buffer name, so the display buffer cannot crowd out the render buffer. */
  const string readback_name = mem.log_name();
  if (!mem.host_pointer || mem.memory_size() == 0 || readback_counts_[readback_name] >= 12) {
    return;
  }
  readback_counts_[readback_name]++;
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

void MetalDeviceQueue::log_state_summary()
{
  /* The render buffer (what the integrator accumulates) and the queue counters (how many paths each kernel was given). */
  if (n_state_logs_ >= 30) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
  bool logged = false;
  for (auto &it : metal_device_->metal_mem_map) {
    device_memory *mem = it.second->mem;
    if (!mem || !it.second->hostPtr) {
      continue;
    }
    const string name = mem->log_name();
    if (name.find("RenderBuffers") != string::npos && mem->data_type == TYPE_FLOAT) {
      log_readback(*mem);
      logged = true;
    }
    else if (name.find("queue_counter") != string::npos) {
      const int *v = (const int *)it.second->hostPtr;
      const size_t count = std::min<size_t>(it.second->size / sizeof(int), 20);
      string text;
      for (size_t i = 0; i < count; i++) {
        text += string_printf(" %d", v[i]);
      }
      fprintf(stderr, "[cycles-wm] \"%s\" first values:%s\n", name.c_str(), text.c_str());
      logged = true;
    }
  }
  if (logged) {
    n_state_logs_++;
  }
}

/* Which buffer an address points into ("name + offset"), for the logs. */
string MetalDeviceQueue::describe_address(uint64_t address)
{
  if (address == 0) {
    return "(null)";
  }
  std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
  for (auto &it : metal_device_->metal_mem_map) {
    const wm::Buffer &b = it.second->buffer;
    if (b.gpu_address && address >= b.gpu_address && address < b.gpu_address + b.size) {
      const string nm = it.second->mem ? it.second->mem->log_name() : string("?");
      return string_printf("-> \"%s\" + %llu (buffer of %llu bytes)",
                           nm.c_str(),
                           (unsigned long long)(address - b.gpu_address),
                           (unsigned long long)b.size);
    }
  }
  const wm::Buffer &lb = metal_device_->launch_params_buffer;
  if (lb.gpu_address && address >= lb.gpu_address && address < lb.gpu_address + lb.size) {
    return "-> launch params";
  }
  return "-> NO BUFFER (not an address the device knows)";
}

/* After the first init_from_camera the queue counters read zero. This puts a one-line kernel of our own on the same queue
 * and has it write into the real queue-counter buffer: if the numbers arrive, the GPU can write that buffer and the
 * Cycles kernels are what leaves it empty. If not, the buffer is not reachable the way the self-test's buffers are. */
void MetalDeviceQueue::probe_counter_write()
{
  probe_done_ = true;
  MetalDevice::MetalMem *target = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
    for (auto &it : metal_device_->metal_mem_map) {
      device_memory *mem = it.second->mem;
      if (mem && it.second->hostPtr && mem->log_name().find("queue_counter") != string::npos) {
        target = it.second.get();
        break;
      }
    }
  }
  if (!target || target->size < 16) {
    fprintf(stderr, "[cycles-wm] probe: no queue-counter buffer\n");
    return;
  }
  volatile unsigned *v = (volatile unsigned *)target->hostPtr;
  const size_t count = std::min<size_t>(target->size / sizeof(int), 20);
  for (size_t i = 0; i < count; i++) {
    if (v[i] != 0) {
      return; /* the counters moved: nothing to probe */
    }
  }
  const size_t last = target->size / sizeof(int) - 1;
  const unsigned saved_last = v[last], saved_prev = v[last - 1];

  const char *src =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void wm_probe(constant ulong *args [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
      "  device atomic_uint *c = (device atomic_uint *)args[0];\n"
      "  device uint *p = (device uint *)args[1];\n"
      "  atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);\n"
      "  if (tid == 0) { p[0] = 0xABCD1234u; }\n"
      "}\n";
  obj_handle_t error = 0;
  obj_handle_t lib = MTLDevice_newLibraryWithSource(metal_device_->mtlDevice, src, strlen(src), 0x0302, 0, &error);
  string text = wm::error_text(error);
  if (!lib) {
    fprintf(stderr, "[cycles-wm] probe: library did not compile: %s\n", text.c_str());
    return;
  }
  obj_handle_t fn = MTLLibrary_newFunction(lib, "wm_probe");
  WMTComputePipelineInfoEx info = {};
  info.compute_function = fn;
  info.max_total_threads_per_threadgroup = 64;
  info.flags = 1u;
  info.immutable_buffers = 0;
  error = 0;
  obj_handle_t pso = MTLDevice_newComputePipelineStateEx(metal_device_->mtlDevice, &info, &error);
  text = wm::error_text(error);
  if (!pso) {
    fprintf(stderr, "[cycles-wm] probe: pipeline did not build: %s\n", text.c_str());
    wm::release(fn);
    wm::release(lib);
    return;
  }

  uint64_t args[2] = {target->buffer.gpu_address + last * sizeof(int),
                      target->buffer.gpu_address + (last - 1) * sizeof(int)};
  obj_handle_t cb = MTLCommandQueue_commandBuffer(mtlCommandQueue_);
  obj_handle_t enc = MTLCommandBuffer_computeCommandEncoder(cb, false);
  wmtcmd_compute_setpso set_pso = {};
  set_pso.type = WMTComputeCommandSetPSO;
  set_pso.pso = pso;
  set_pso.threadgroup_size = {64, 1, 1};
  wmtcmd_compute_setbytes set_args = {};
  set_args.type = WMTComputeCommandSetBytes;
  set_args.bytes.ptr = args;
  set_args.length = sizeof(args);
  set_args.index = 0;
  wmtcmd_compute_dispatch dispatch = {};
  dispatch.type = WMTComputeCommandDispatchThreads;
  dispatch.size = {16, 1, 1};
  set_pso.next.ptr = &set_args;
  set_args.next.ptr = &dispatch;
  dispatch.next.ptr = nullptr;
  MTLComputeCommandEncoder_encodeCommands(enc, (wmtcmd_base *)&set_pso);
  MTLCommandEncoder_endEncoding(enc);
  MTLCommandBuffer_commit(cb);
  MTLCommandBuffer_waitUntilCompleted(cb);
  const int status = int(MTLCommandBuffer_status(cb));
  fprintf(stderr,
          "[cycles-wm] probe: the queue counters were all zero after init_from_camera; own kernel on the real counter buffer: "
          "status %d, atomic word %u (16 expected), plain store 0x%X (0xABCD1234 expected) -> %s\n",
          status,
          v[last],
          (unsigned)v[last - 1],
          (v[last] == saved_last + 16 && v[last - 1] == 0xABCD1234u) ? "THE GPU CAN WRITE THIS BUFFER" :
                                                                       "THE GPU CANNOT WRITE THIS BUFFER");
  v[last] = saved_last;
  v[last - 1] = saved_prev;
  wm::release(pso);
  wm::release(fn);
  wm::release(lib);
}

/* Compares what the host wrote into the launch parameters with what the GPU reads back from them (same sizes, same offsets,
 * same contents?). A Windows build lays structs out by its own rules; the Metal compiler uses Apple's. */
void MetalDeviceQueue::layout_check()
{
  layout_done_ = true;
  MetalDevice *d = metal_device_;
  if (!d->mtlLibrary || !d->launch_params) {
    return;
  }
  obj_handle_t fn = MTLLibrary_newFunction(d->mtlLibrary, "wm_layout");
  if (!fn) {
    fprintf(stderr, "[cycles-wm] layout: the library has no wm_layout function\n");
    return;
  }
  WMTComputePipelineInfoEx info = {};
  info.compute_function = fn;
  info.max_total_threads_per_threadgroup = 1;
  info.flags = 0;
  info.immutable_buffers = 0;
  obj_handle_t error = 0;
  obj_handle_t pso = MTLDevice_newComputePipelineStateEx(d->mtlDevice, &info, &error);
  const string text = wm::error_text(error);
  if (!pso) {
    fprintf(stderr, "[cycles-wm] layout: pipeline did not build: %s\n", text.c_str());
    wm::release(fn);
    return;
  }
  wm::Buffer out = wm::new_buffer(d->mtlDevice, 4096);
  d->metal_mem_alloc(out);
  d->prepare_residency();
  memset(out.contents, 0xEE, 4096);

  obj_handle_t cb = MTLCommandQueue_commandBuffer(mtlCommandQueue_);
  obj_handle_t enc = MTLCommandBuffer_computeCommandEncoder(cb, false);
  wmtcmd_compute_setpso set_pso = {};
  set_pso.type = WMTComputeCommandSetPSO;
  set_pso.pso = pso;
  set_pso.threadgroup_size = {1, 1, 1};
  wmtcmd_compute_setbuffer b0 = {};
  b0.type = WMTComputeCommandSetBuffer;
  b0.buffer = out.handle;
  b0.offset = 0;
  b0.index = 0;
  wmtcmd_compute_setbuffer b1 = {};
  b1.type = WMTComputeCommandSetBuffer;
  b1.buffer = d->launch_params_buffer.handle;
  b1.offset = 0;
  b1.index = 1;
  wmtcmd_compute_dispatch dispatch = {};
  dispatch.type = WMTComputeCommandDispatchThreads;
  dispatch.size = {1, 1, 1};
  set_pso.next.ptr = &b0;
  b0.next.ptr = &b1;
  b1.next.ptr = &dispatch;
  dispatch.next.ptr = nullptr;
  MTLComputeCommandEncoder_encodeCommands(enc, (wmtcmd_base *)&set_pso);
  MTLCommandEncoder_endEncoding(enc);
  MTLCommandBuffer_commit(cb);
  MTLCommandBuffer_waitUntilCompleted(cb);
  const int status = int(MTLCommandBuffer_status(cb));

  const uint32_t *g = (const uint32_t *)out.contents;
  const uint32_t host_sizes[5] = {uint32_t(sizeof(KernelParamsMetal)),
                                  uint32_t(sizeof(KernelData)),
                                  uint32_t(sizeof(IntegratorStateGPU)),
                                  uint32_t(offsetof(KernelParamsMetal, data)),
                                  uint32_t(offsetof(KernelParamsMetal, integrator_state))};
  const char *labels[5] = {"sizeof(KernelParamsMetal)", "sizeof(KernelData)", "sizeof(IntegratorStateGPU)", "offsetof(data)", "offsetof(integrator_state)"};
  int bad = 0;
  for (int i = 0; i < 5; i++) {
    fprintf(stderr, "[cycles-wm] layout: %s host %u, GPU %u %s\n", labels[i], host_sizes[i], g[i], host_sizes[i] == g[i] ? "ok" : "<-- DIFFERENT");
    bad += host_sizes[i] != g[i];
  }
  const uint32_t *dp = (const uint32_t *)&d->launch_params->data;
  const uint32_t *sp = (const uint32_t *)&d->launch_params->integrator_state;
  int first_bad_data = -1, first_bad_state = -1;
  for (uint32_t c = 0; c < 16; c++) {
    uint32_t h = 0, h2 = 0;
    for (uint32_t i = c * 256; i < (c + 1) * 256 && i < sizeof(KernelData) / 4; i++) {
      h = h * 31u + dp[i];
    }
    for (uint32_t i = c * 64; i < (c + 1) * 64 && i < sizeof(IntegratorStateGPU) / 4; i++) {
      h2 = h2 * 31u + sp[i];
    }
    if (h != g[8 + c] && first_bad_data < 0) {
      first_bad_data = int(c);
    }
    if (h2 != g[24 + c] && first_bad_state < 0) {
      first_bad_state = int(c);
    }
  }
  fprintf(stderr,
          "[cycles-wm] layout: command buffer status %d; KernelData contents %s (first differing 1 KB block %d); IntegratorStateGPU contents %s (first differing 256 B block %d) -> %s\n",
          status,
          first_bad_data < 0 ? "identical" : "DIFFERENT",
          first_bad_data,
          first_bad_state < 0 ? "identical" : "DIFFERENT",
          first_bad_state,
          (bad == 0 && first_bad_data < 0 && first_bad_state < 0) ? "THE GPU SEES WHAT THE HOST WROTE" : "LAYOUT MISMATCH");
  d->metal_mem_free(out.handle, out.size);
  wm::release(pso);
  wm::release(fn);
}

bool MetalDeviceQueue::enqueue(DeviceKernel kernel,
                               const int work_size,
                               const DeviceKernelArguments &args)
{
  if (metal_device_->have_error()) {
    return false;
  }

  const double enqueue_started = time_dt();
  debug_enqueue_begin(kernel, work_size);
  n_enqueue_++;
  last_kernel_ = kernel;
  if (!layout_done_ && kernel == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA) {
    layout_check();
  }
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

  if (dispatch_logs_[kernel] < 2) {
    dispatch_logs_[kernel]++;
    fprintf(stderr,
            "[cycles-wm] dispatch %s: work size %d, threadgroup %d, shared %d bytes, %zu argument bytes\n",
            device_kernel_as_string(kernel),
            work_size,
            num_threads_per_block,
            shared_mem_bytes,
            dynamic_bytes_written);
    /* The arguments as 64-bit words, with the buffer each one points into. */
    for (size_t off = 0; off + 8 <= dynamic_bytes_written; off += 8) {
      uint64_t w;
      memcpy(&w, dynamic_args + off, 8);
      fprintf(stderr, "[cycles-wm]   arg word %zu = 0x%llx %s\n", off / 8, (unsigned long long)w, describe_address(w).c_str());
    }
    if (kernel == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA) {
      {
        std::lock_guard<std::recursive_mutex> lock(metal_device_->metal_mem_map_mutex);
        for (auto &it : metal_device_->metal_mem_map) {
          if (it.second->mem && it.second->hostPtr && it.second->mem->log_name() == "work_tiles") {
            const int *t = (const int *)it.second->hostPtr;
            string text;
            for (size_t i = 0; i < std::min<size_t>(it.second->size / 4, 11); i++) {
              text += string_printf(" %d", t[i]);
            }
            fprintf(stderr, "[cycles-wm]   work_tiles ints:%s\n", text.c_str());
          }
        }
      }
      const uint64_t *lp = (const uint64_t *)&metal_device_->launch_params->integrator_state;
      const size_t n = offsetof(IntegratorStateGPU, sort_partition_divisor) / 8;
      for (size_t i = 0; i < n; i++) {
        fprintf(stderr, "[cycles-wm]   launch integrator_state[%zu] = 0x%llx %s\n", i, (unsigned long long)lp[i], describe_address(lp[i]).c_str());
      }
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
  encode_seconds_ += time_dt() - enqueue_started;

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
    const double wait_started = time_dt();
    MTLCommandBuffer_commit(mtlCommandBuffer_);
    MTLCommandBuffer_waitUntilCompleted(mtlCommandBuffer_);
    wait_seconds_ += time_dt() - wait_started;

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
  log_state_summary();
  if (!probe_done_ && last_kernel_ == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA) {
    probe_counter_write();
  }

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
  pending_readbacks_.push_back(&mem);
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
