/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "device/metal_wm/device_impl.h"
#  include "device/metal_wm/queue.h"

#  include "scene/scene.h"

#  include "util/debug.h"
#  include "util/log.h"
#  include "util/md5.h"
#  include "util/path.h"
#  include "util/time.h"

CCL_NAMESPACE_BEGIN

void MetalDevice::set_error(const string &error)
{
  static std::mutex s_error_mutex;
  std::lock_guard<std::mutex> lock(s_error_mutex);

  Device::set_error(error);

  if (!has_error) {
    fprintf(stderr, "\nRefer to the Cycles GPU rendering documentation for possible solutions:\n");
    fprintf(stderr,
            "https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html\n\n");
    has_error = true;
  }
}

MetalDevice::MetalDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
    : Device(info, stats, profiler, headless), image_info(this, "image_info", MEM_GLOBAL)
{
  const vector<obj_handle_t> &all = wm::devices();
  assert(info.num >= 0 && info.num < (int)all.size());
  mtlDevice = all[info.num];

  apple9 = MTLDevice_supportsFamily(mtlDevice, WMTGPUFamilyApple9);
  use_metalrt = false;

  mtlComputeCommandQueue = MTLDevice_newCommandQueue(mtlDevice, 64);

  /* One residency set for everything the GPU reads or writes (macOS 15 / iOS 18 and later). */
  mtlResidencySet = MTLDevice_newResidencySet(mtlDevice, 512);
  if (mtlResidencySet) {
    MTLCommandQueue_addResidencySet(mtlComputeCommandQueue, mtlResidencySet);
  }

  image_bindings = wm::new_buffer(mtlDevice, 8192);
  metal_mem_alloc(image_bindings);
  if (image_bindings.contents) {
    memset(image_bindings.contents, 0, image_bindings.size);
  }

  launch_params_buffer = wm::new_buffer(mtlDevice, sizeof(KernelParamsMetal));
  metal_mem_alloc(launch_params_buffer);
  launch_params = (KernelParamsMetal *)launch_params_buffer.contents;
  if (launch_params) {
    memset(launch_params, 0, sizeof(KernelParamsMetal));
  }
  prepare_residency();
}

MetalDevice::~MetalDevice()
{
  image_info.free();
  for (MetalPipeline &p : pipelines) {
    wm::release(p.pipeline);
    wm::release(p.function);
  }
  metal_mem_free(launch_params_buffer.handle, launch_params_buffer.size);
  metal_mem_free(image_bindings.handle, image_bindings.size);
  flush_delayed_free_list();
  wm::release(mtlLibrary);
  wm::release(mtlResidencySet);
  /* The command queue and the device belong to the process. */
}

size_t MetalDevice::recommended_working_set() const
{
  return MTLDevice_recommendedMaxWorkingSetSize(mtlDevice);
}

void MetalDevice::metal_mem_alloc(const wm::Buffer &buffer)
{
  if (buffer.handle) {
    stats.mem_alloc(buffer.size);
    if (mtlResidencySet) {
      std::lock_guard<std::mutex> lock(residency_mutex);
      MTLResidencySet_addAllocation(mtlResidencySet, buffer.handle);
      residency_dirty = true;
    }
  }
}

void MetalDevice::metal_mem_free(obj_handle_t &handle, const size_t size)
{
  if (handle) {
    stats.mem_free(size);
    std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
    if (mtlResidencySet) {
      std::lock_guard<std::mutex> rlock(residency_mutex);
      MTLResidencySet_removeAllocation(mtlResidencySet, handle);
      residency_dirty = true;
    }
    /* Freed after the next synchronize, when the GPU is done with it. */
    delayed_free_list.push_back(handle);
    handle = 0;
  }
}

void MetalDevice::flush_delayed_free_list()
{
  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
  for (obj_handle_t handle : delayed_free_list) {
    NSObject_release(handle);
  }
  delayed_free_list.clear();
}

void MetalDevice::prepare_residency()
{
  if (mtlResidencySet) {
    std::lock_guard<std::mutex> lock(residency_mutex);
    if (residency_dirty) {
      residency_dirty = false;
      MTLResidencySet_commit(mtlResidencySet);
    }
  }
}

/* ---- kernels ---- */

string MetalDevice::preprocess_source()
{
  string global_defines;

  if (DebugFlags().metal.adaptive_compile) {
    global_defines += "#define __KERNEL_FEATURES__ " + to_string(kernel_features) + "\n";
  }
  if (DebugFlags().metal.use_local_atomic_sort) {
    global_defines += "#define __KERNEL_LOCAL_ATOMIC_SORT__\n";
  }
  global_defines += "#define __KERNEL_METAL_APPLE__\n";
  global_defines += "#define __METAL_GLOBAL_BUILTINS__\n";
  global_defines += "#define __KERNEL_METAL_MACOS__ 27\n";
  global_defines += "#define __KERNEL_METAL_TARGET_CPU_ARM64__\n";

  return global_defines;
}

bool MetalDevice::compile_library()
{
  string text = "\n#include \"kernel/device/metal/kernel.metal\"\n";
  text = path_source_replace_includes(text, path_get("source"));
  source = preprocess_source() + text;

  const string key = preprocess_source();
  if (mtlLibrary && key == library_features_key) {
    return true;
  }
  library_features_key = key;
  wm::release(mtlLibrary);

  const double starttime = time_dt();
  obj_handle_t error = 0;
  mtlLibrary = MTLDevice_newLibraryWithSource(
      mtlDevice, source.data(), source.size(), 0x0302, WMT_LIBRARY_FLAG_FAST_MATH, &error);
  const string text_error = wm::error_text(error);
  LOG_INFO << "Front-end compilation finished in " << (time_dt() - starttime) << " seconds";

  if (!mtlLibrary) {
    set_error(string_printf("Failed to compile library:\n%s", text_error.c_str()));
    return false;
  }
  /* The old pipelines belong to the old library. */
  for (MetalPipeline &p : pipelines) {
    wm::release(p.pipeline);
    wm::release(p.function);
    p = MetalPipeline();
  }
  return true;
}

bool MetalDevice::load_kernels(const uint _kernel_features)
{
  kernel_features |= _kernel_features;
  return compile_library();
}

bool MetalDevice::is_ready(string &status) const
{
  if (!error_msg.empty()) {
    return true;
  }
  status = "";
  return mtlLibrary != 0;
}

const MetalPipeline *MetalDevice::get_pipeline(const DeviceKernel kernel)
{
  MetalPipeline &p = pipelines[kernel];
  if (p.pipeline) {
    return &p;
  }
  if (p.failed || !mtlLibrary) {
    return nullptr;
  }

  const string function_name = string("cycles_metal_") + device_kernel_as_string(kernel);
  p.function = MTLLibrary_newFunction(mtlLibrary, function_name.c_str());
  if (!p.function) {
    p.failed = true;
    set_error(string_printf("Could not find the Metal function \"%s\"", function_name.c_str()));
    return nullptr;
  }

  /* Cycles tunes some GPUs for 64 threads per threadgroup; the others use what the pipeline allows. */
  int threads_per_threadgroup = max_threads_per_threadgroup;
  int num_threads_per_block = 0;
  if (apple9) {
    threads_per_threadgroup = 64;
    num_threads_per_block = 64;
  }

  WMTComputePipelineInfoEx info = {};
  info.compute_function = p.function;
  info.max_total_threads_per_threadgroup = threads_per_threadgroup;
  info.flags = 1u;              /* the threadgroup size is a multiple of the execution width */
  info.immutable_buffers = 0x7; /* buffers 0, 1 and 2 are never written by the kernels */
  obj_handle_t error = 0;
  const double starttime = time_dt();
  p.pipeline = MTLDevice_newComputePipelineStateEx(mtlDevice, &info, &error);
  const string text_error = wm::error_text(error);
  if (!p.pipeline) {
    p.failed = true;
    set_error(string_printf("Failed to build the pipeline for %s:\n%s",
                            device_kernel_as_string(kernel),
                            text_error.c_str()));
    return nullptr;
  }
  LOG_TRACE << "Pipeline " << device_kernel_as_string(kernel) << " built in "
            << (time_dt() - starttime) << " s";

  if (!num_threads_per_block) {
    WMTComputePipelineLimits limits = {};
    MTLComputePipelineState_limits(p.pipeline, &limits);
    num_threads_per_block = round_down(limits.max_total_threads_per_threadgroup,
                                       limits.thread_execution_width);
    num_threads_per_block = std::max(num_threads_per_block, (int)limits.thread_execution_width);
  }
  p.num_threads_per_block = num_threads_per_block;
  return &p;
}

unique_ptr<DeviceQueue> MetalDevice::gpu_queue_create()
{
  return make_unique<MetalDeviceQueue>(this);
}

/* ---- memory ---- */

void MetalDevice::erase_allocation(device_memory &mem)
{
  mem.device_pointer = 0;
  mem.device_size = 0;

  auto it = metal_mem_map.find(&mem);
  if (it != metal_mem_map.end()) {
    MetalMem *mmem = it->second.get();

    /* blank out reference to MetalMem* in the launch params (fixes crash #94736) */
    if (mmem->pointer_index >= 0) {
      device_ptr *pointers = (device_ptr *)launch_params;
      pointers[mmem->pointer_index] = 0;
    }
    metal_mem_map.erase(it);
  }
}

MetalDevice::MetalMem *MetalDevice::generic_alloc(device_memory &mem)
{
  const size_t size = mem.memory_size();
  mem.device_pointer = 0;

  wm::Buffer buffer;
  if (size > 0) {
    /* Unified memory: every buffer is shared, so the host side is just the buffer's contents. */
    buffer = wm::new_buffer(mtlDevice, size, true);
    if (!buffer.handle) {
      set_error("System is out of GPU memory");
      return nullptr;
    }
  }

  LOG_DEBUG << "Buffer allocate: " << mem.log_name() << ", " << string_human_readable_number(size)
            << " bytes";

  mem.device_size = size;
  metal_mem_alloc(buffer);

  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);

  assert(metal_mem_map.count(&mem) == 0); /* assert against double-alloc */
  unique_ptr<MetalMem> mmem = make_unique<MetalMem>();

  mmem->mem = &mem;
  mmem->buffer = buffer;
  mmem->offset = 0;
  mmem->size = size;
  mmem->hostPtr = buffer.contents;

  /* Each device_memory object points at its MetalMem; the kernel parameters get the buffer's GPU address in const_copy_to. */
  mem.device_pointer = device_ptr(mmem.get());

  if (mem.host_pointer && mem.host_pointer != mmem->hostPtr && mmem->hostPtr) {
    memcpy(mmem->hostPtr, mem.host_pointer, size);
    host_free(mem.type, mem.host_pointer, mem.memory_size());
    mem.host_pointer = mmem->hostPtr;
  }
  mem.shared_pointer = mmem->hostPtr;
  mem.shared_counter++;

  MetalMem *mmem_ptr = mmem.get();
  metal_mem_map[&mem] = std::move(mmem);

  if (stats.mem_used > recommended_working_set()) {
    set_error("System is out of GPU memory");
    return nullptr;
  }

  return mmem_ptr;
}

void MetalDevice::generic_free(device_memory &mem)
{
  if (!mem.device_pointer) {
    return;
  }

  assert(mem.host_pointer == nullptr || mem.host_pointer == mem.shared_pointer);

  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
  MetalMem &mmem = *metal_mem_map.at(&mem);
  const size_t size = mmem.size;

  bool free_buffer = true;
  if (mem.shared_pointer) {
    assert(mem.shared_counter > 0);
    if (--mem.shared_counter > 0) {
      free_buffer = false;
    }
  }

  if (free_buffer) {
    if (mem.host_pointer && mem.host_pointer == mem.shared_pointer) {
      /* The host side lived in the buffer: give the host its own copy back. */
      mem.host_pointer = mem.host_alloc(size);
      memcpy(mem.host_pointer, mem.shared_pointer, size);
    }
    mem.shared_pointer = nullptr;

    metal_mem_free(mmem.buffer.handle, size);
  }

  erase_allocation(mem);
}

void MetalDevice::mem_alloc(device_memory &mem)
{
  if (mem.type == MEM_IMAGE_TEXTURE) {
    assert(!"mem_alloc not supported for images.");
  }
  else if (mem.type == MEM_GLOBAL) {
    generic_alloc(mem);
  }
  else {
    generic_alloc(mem);
  }
}

void MetalDevice::mem_copy_to(device_memory &mem)
{
  if (!mem.device_pointer) {
    if (mem.type == MEM_GLOBAL) {
      global_alloc(mem);
    }
    else if (mem.type == MEM_IMAGE_TEXTURE) {
      image_alloc((device_image &)mem);
    }
    else {
      generic_alloc(mem);
    }
  }
  /* Shared buffers: the host side already is the buffer. */
}

void MetalDevice::mem_move_to_host(device_memory & /*mem*/)
{
  assert(!"Metal does not support mem_move_to_host");
}

void MetalDevice::mem_copy_from(
    device_memory & /*mem*/, const size_t /*y*/, size_t /*w*/, const size_t /*h*/, size_t /*elem*/)
{
  /* Shared buffers: nothing to copy (the queue synchronizes before the host reads). */
}

void MetalDevice::mem_or_from_device(device_memory & /*mem*/) {}

void MetalDevice::mem_zero(device_memory &mem)
{
  if (!mem.device_pointer) {
    mem_alloc(mem);
  }
  if (mem.shared_pointer) {
    memset(mem.shared_pointer, 0, mem.memory_size());
  }
}

void MetalDevice::mem_free(device_memory &mem)
{
  if (mem.type == MEM_GLOBAL) {
    global_free(mem);
  }
  else if (mem.type == MEM_IMAGE_TEXTURE) {
    image_free((device_image &)mem);
  }
  else {
    generic_free(mem);
  }
}

device_ptr MetalDevice::mem_alloc_sub_ptr(device_memory & /*mem*/,
                                          size_t /*offset*/,
                                          size_t /*size*/)
{
  assert(0);
  return 0;
}

void MetalDevice::const_copy_to(const char *name, void *host, const size_t size)
{
  if (strcmp(name, "data") == 0) {
    assert(size == sizeof(KernelData));
    memcpy((uint8_t *)&launch_params->data, host, sizeof(KernelData));
    return;
  }

  /* Update data storage pointers in launch parameters. */
  auto update_launch_pointers = [&](size_t offset, void *data, const size_t pointers_size) {
    uint64_t *addresses = (uint64_t *)((uint8_t *)launch_params + offset);

    MetalMem **mmem = (MetalMem **)data;
    int pointer_count = pointers_size / sizeof(device_ptr);
    int pointer_index = offset / sizeof(device_ptr);
    for (int i = 0; i < pointer_count; i++) {
      addresses[i] = 0;
      if (mmem[i]) {
        mmem[i]->pointer_index = pointer_index + i;
        if (mmem[i]->buffer.handle) {
          addresses[i] = mmem[i]->buffer.gpu_address;
        }
      }
    }
  };

  /* Update data storage pointers in launch parameters. */
  if (strcmp(name, "integrator_state") == 0) {
    /* IntegratorStateGPU is contiguous pointers up to sort_partition_divisor */
    const size_t pointer_block_size = offsetof(IntegratorStateGPU, sort_partition_divisor);
    update_launch_pointers(
        offsetof(KernelParamsMetal, integrator_state), host, pointer_block_size);

    /* Ensure the non-pointer part of IntegratorStateGPU is copied (sort_partition_divisor etc). */
    memcpy((uint8_t *)&launch_params->integrator_state + pointer_block_size,
           (uint8_t *)host + pointer_block_size,
           sizeof(IntegratorStateGPU) - pointer_block_size);
  }
#  define KERNEL_DATA_ARRAY(data_type, tex_name) \
    else if (strcmp(name, #tex_name) == 0) { \
      update_launch_pointers(offsetof(KernelParamsMetal, tex_name), host, size); \
    }
#  include "kernel/data_arrays.h"
#  undef KERNEL_DATA_ARRAY
}

void MetalDevice::global_alloc(device_memory &mem)
{
  if (mem.is_resident(this)) {
    generic_alloc(mem);
  }

  const_copy_to(mem.global_name(), &mem.device_pointer, sizeof(mem.device_pointer));
}

void MetalDevice::global_free(device_memory &mem)
{
  if (mem.is_resident(this) && mem.device_pointer) {
    generic_free(mem);
  }
}

void MetalDevice::image_alloc(device_image &mem)
{
  /* Textures are not in this port yet. */
  (void)mem;
  set_error("Image textures are not supported by this Metal device yet");
}

void MetalDevice::image_free(device_image &mem)
{
  if (mem.device_pointer) {
    generic_free(mem);
  }
}

void MetalDevice::load_image_info()
{
  image_info.copy_to_device();
  if (image_bindings.contents) {
    memset(image_bindings.contents, 0, image_bindings.size);
  }
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
