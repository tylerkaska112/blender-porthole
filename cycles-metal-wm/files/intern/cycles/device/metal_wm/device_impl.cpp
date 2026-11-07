/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  ifdef _WIN32
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#  endif

#  include <algorithm>
#  include <mutex>
#  include <vector>

#  include "device/metal_wm/device_impl.h"
#  include "device/metal_wm/queue.h"

#  include "scene/scene.h"

#  include "util/debug.h"
#  include "util/log.h"
#  include "util/md5.h"
#  include "util/path.h"
#  include "util/time.h"

CCL_NAMESPACE_BEGIN

static const std::vector<DeviceKernel> &core_kernel_list();

/* Every MetalDevice that is alive: the viewport and a final render are two devices on one GPU and one process memory limit. */
static std::mutex g_devices_mutex;
static std::vector<const MetalDevice *> g_devices;

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
  {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    g_devices.push_back(this);
  }

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

  self_test();

  fprintf(stderr,
          "[cycles-wm] device \"%s\", working set %zu MB (Metal reports %zu MB)\n",
          wm::device_name(mtlDevice).c_str(),
          recommended_working_set() >> 20,
          size_t(MTLDevice_recommendedMaxWorkingSetSize(mtlDevice)) >> 20);
}

MetalDevice::~MetalDevice()
{
  {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    g_devices.erase(std::remove(g_devices.begin(), g_devices.end(), this), g_devices.end());
  }
  join_precompile();
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
  /* CYCLES_METAL_WORKING_SET_MB sets the budget outright. */
  if (const char *str = getenv("CYCLES_METAL_WORKING_SET_MB")) {
    const long mb = atol(str);
    if (mb > 0) {
      return size_t(mb) << 20;
    }
  }

  size_t budget = MTLDevice_recommendedMaxWorkingSetSize(mtlDevice);

#  ifdef _WIN32
  /* Under Porthole this number is the video memory a Windows GAME can afford: the process limit minus the
   * RAM the game is told it has, minus the runtime's own cost, with a floor of 1 GB (seen on an iPhone 15
   * Pro: 1024 MB while Metal itself recommends 5461 MB). Cycles keeps nothing in guest RAM, so it gets at
   * least 2.5 GB; the process limit stays the real ceiling and CYCLES_METAL_WORKING_SET_MB moves it. */
  const size_t floor_budget = size_t(2560) << 20;
  if (budget < floor_budget) {
    budget = floor_budget;
  }
#  endif
  return budget;
}

void MetalDevice::self_test()
{
  static std::once_flag once;
  std::call_once(once, [this]() {
    fprintf(stderr, "[cycles-wm] selftest: residency set %s\n", mtlResidencySet ? "in use" : "NOT available (resources are named per dispatch)");

    const char *src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void wm_selftest(constant ulong *args [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
        "  device const float *in = (device const float *)args[0];\n"
        "  device float *out = (device float *)args[1];\n"
        "  device atomic_uint *counter = (device atomic_uint *)args[2];\n"
        "  out[tid] = in[tid] * 2.0f + 1.0f;\n"
        "  atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);\n"
        "}\n";
    obj_handle_t error = 0;
    obj_handle_t lib = MTLDevice_newLibraryWithSource(mtlDevice, src, strlen(src), 0x0302, 0, &error);
    string text = wm::error_text(error);
    if (!lib) {
      fprintf(stderr, "[cycles-wm] selftest: the test library did not compile: %s\n", text.c_str());
      return;
    }
    obj_handle_t fn = MTLLibrary_newFunction(lib, "wm_selftest");
    WMTComputePipelineInfoEx info = {};
    info.compute_function = fn;
    info.max_total_threads_per_threadgroup = 64;
    info.flags = 1u;
    info.immutable_buffers = 0;
    error = 0;
    obj_handle_t pso = MTLDevice_newComputePipelineStateEx(mtlDevice, &info, &error);
    text = wm::error_text(error);
    if (!pso) {
      fprintf(stderr, "[cycles-wm] selftest: the test pipeline did not build: %s\n", text.c_str());
      wm::release(fn);
      wm::release(lib);
      return;
    }

    const int n = 4096;
    wm::Buffer in = wm::new_buffer(mtlDevice, n * sizeof(float));
    wm::Buffer out = wm::new_buffer(mtlDevice, n * sizeof(float));
    wm::Buffer counter = wm::new_buffer(mtlDevice, 16);
    if (!in.handle || !out.handle || !counter.handle) {
      fprintf(stderr, "[cycles-wm] selftest: could not allocate the test buffers\n");
      return;
    }
    metal_mem_alloc(in);
    metal_mem_alloc(out);
    metal_mem_alloc(counter);
    prepare_residency();
    for (int i = 0; i < n; i++) {
      ((float *)in.contents)[i] = float(i);
      ((float *)out.contents)[i] = -1.0f;
    }
    memset(counter.contents, 0, 16);

    uint64_t args[4] = {in.gpu_address, out.gpu_address, counter.gpu_address, 0};

    obj_handle_t cb = MTLCommandQueue_commandBuffer(mtlComputeCommandQueue);
    obj_handle_t enc = MTLCommandBuffer_computeCommandEncoder(cb, false);

    vector<wmtcmd_compute_useresource> uses;
    if (!mtlResidencySet) {
      for (const obj_handle_t r : {in.handle, out.handle, counter.handle}) {
        wmtcmd_compute_useresource use = {};
        use.type = WMTComputeCommandUseResource;
        use.resource = r;
        use.usage = WMTResourceUsage(WMTResourceUsageRead | WMTResourceUsageWrite);
        uses.push_back(use);
      }
    }
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
    dispatch.size = {(uint64_t)n, 1, 1};

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
    link(&set_args);
    link(&dispatch);
    MTLComputeCommandEncoder_encodeCommands(enc, head);
    MTLCommandEncoder_endEncoding(enc);
    MTLCommandBuffer_commit(cb);
    MTLCommandBuffer_waitUntilCompleted(cb);
    const int status = int(MTLCommandBuffer_status(cb));

    int good = 0;
    for (int i = 0; i < n; i++) {
      if (((float *)out.contents)[i] == float(i) * 2.0f + 1.0f) {
        good++;
      }
    }
    const unsigned counted = *(volatile unsigned *)counter.contents;
    fprintf(stderr,
            "[cycles-wm] selftest: command buffer status %d, %d of %d outputs correct, atomic counter %u (expected %d) -> %s\n",
            status,
            good,
            n,
            counted,
            n,
            (good == n && counted == (unsigned)n) ? "GPU PATH OK" : "GPU PATH BROKEN");

    metal_mem_free(in.handle, in.size);
    metal_mem_free(out.handle, out.size);
    metal_mem_free(counter.handle, counter.size);
    wm::release(pso);
    wm::release(fn);
    wm::release(lib);
  });
}

size_t MetalDevice::total_mem_used() const
{
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  size_t total = 0;
  for (const MetalDevice *device : g_devices) {
    total += device->stats.mem_used;
  }
  return max(total, size_t(stats.mem_used));
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

void MetalDevice::metal_mem_alloc(obj_handle_t handle, const size_t size)
{
  if (handle) {
    stats.mem_alloc(size);
    if (mtlResidencySet) {
      std::lock_guard<std::mutex> lock(residency_mutex);
      MTLResidencySet_addAllocation(mtlResidencySet, handle);
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
  return preprocess_source_for(kernel_features);
}

string MetalDevice::preprocess_source_for(const uint features)
{
  string global_defines;

  /* Kernels specialized for what the scene uses compile much faster than the generic ones (the A17 needed
   * 236 s for the generic set). Under Porthole this is the default; CYCLES_METAL_ADAPTIVE_COMPILE=0 turns it off. */
  bool adaptive_compile = DebugFlags().metal.adaptive_compile;
#  ifdef _WIN32
  adaptive_compile = true;
#  endif
  if (const char *env = getenv("CYCLES_METAL_ADAPTIVE_COMPILE")) {
    adaptive_compile = (atoi(env) != 0) || (env[0] != '0');
  }
  if (adaptive_compile) {
    global_defines += "#define __KERNEL_FEATURES__ " + to_string(features) + "\n";
  }
  if (DebugFlags().metal.use_local_atomic_sort) {
    global_defines += "#define __KERNEL_LOCAL_ATOMIC_SORT__\n";
  }
  /* Test knob: a different value makes a source Metal has not compiled before (cold compile timings). */
  if (const char *bust = getenv("CYCLES_WM_CACHE_BUST")) {
    global_defines += string("#define __WM_CACHE_BUST__ ") + bust + "\n";
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
  /* A kernel that reports how the GPU sees the launch parameters (sizes, offsets, a hash of every 256 words), to be
   * compared with what the host wrote (MetalDeviceQueue::layout_check). */
  text += "\nkernel void wm_layout(device uint *out [[buffer(0)]], constant KernelParamsMetal &launch_params_metal [[buffer(1)]], uint tid [[thread_position_in_grid]])\n"
          "{\n"
          "  if (tid != 0) { return; }\n"
          "  constant uint *dp = (constant uint *)&launch_params_metal.data;\n"
          "  constant uint *sp = (constant uint *)&launch_params_metal.integrator_state;\n"
          "  out[0] = uint(sizeof(KernelParamsMetal)); out[1] = uint(sizeof(KernelData)); out[2] = uint(sizeof(IntegratorStateGPU));\n"
          "  out[3] = uint((constant char *)&launch_params_metal.data - (constant char *)&launch_params_metal);\n"
          "  out[4] = uint((constant char *)&launch_params_metal.integrator_state - (constant char *)&launch_params_metal);\n"
          "  for (uint c = 0; c < 16; c++) {\n"
          "    uint h = 0; uint h2 = 0;\n"
          "    for (uint i = c * 256; i < (c + 1) * 256 && i < uint(sizeof(KernelData) / 4); i++) { h = h * 31u + dp[i]; }\n"
          "    for (uint i = c * 64; i < (c + 1) * 64 && i < uint(sizeof(IntegratorStateGPU) / 4); i++) { h2 = h2 * 31u + sp[i]; }\n"
          "    out[8 + c] = h; out[24 + c] = h2;\n"
          "  }\n"
          "}\n";
  source = preprocess_source() + text;
  if (const char *dump = getenv("CYCLES_WM_DUMP_SOURCE")) {
    /* For the kernel pack builder: the exact source this device would compile for these features. */
    if (FILE *f = fopen(string_printf("%s/%08x.metal", dump, kernel_features).c_str(), "wb")) {
      fwrite(source.data(), 1, source.size(), f);
      fclose(f);
    }
  }
  prebuilt_active = false;

  const string key = preprocess_source();
  if (mtlLibrary && key == library_features_key) {
    return true;
  }
  join_precompile();
  library_features_key = key;
  wm::release(mtlLibrary);

  const double starttime = time_dt();
  obj_handle_t error = 0;
  mtlLibrary = MTLDevice_newLibraryWithSource(
      mtlDevice, source.data(), source.size(), 0x0302, WMT_LIBRARY_FLAG_FAST_MATH, &error);
  const string text_error = wm::error_text(error);
  fprintf(stderr, "[cycles-wm] kernel library compiled in %.2f s\n", time_dt() - starttime);

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
  start_precompile();
  return true;
}

static bool read_whole_file(const string &path, vector<uint8_t> &out)
{
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) {
    return false;
  }
  fseek(f, 0, SEEK_END);
  const long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(size > 0 ? size_t(size) : 0);
  const bool ok = size > 0 && fread(out.data(), 1, out.size(), f) == out.size();
  fclose(f);
  return ok;
}

string MetalDevice::kernel_pack_dir()
{
  if (const char *env = getenv("CYCLES_WM_KERNEL_DIR")) {
    return env;
  }
#  ifdef _WIN32
  /* Porthole's Software page installs the pack beside Blender (C:\Program Files\Blender Kernels); a `kernels` folder next to blender.exe works too. Both need Porthole 1.3.41 or
   * later, whose winemetal has the binary archive call (it writes C:\\windows\\porthole-winemetal-slots). */
  {
    vector<uint8_t> marker;
    if (!read_whole_file("C:\\windows\\porthole-winemetal-slots", marker) || atoi(string(marker.begin(), marker.end()).c_str()) < 171) {
      return "";
    }
  }
  char exe[1024] = {0};
  if (GetModuleFileNameA(nullptr, exe, sizeof(exe) - 1)) {
    string dir = exe;
    const size_t cut = dir.find_last_of("\\/");
    if (cut != string::npos) {
      dir.resize(cut);
    }
    vector<uint8_t> probe;
    if (read_whole_file(dir + "\\kernels\\masks.txt", probe)) {
      return dir + "\\kernels";
    }
    const size_t parent = dir.find_last_of("\\/");
    if (parent != string::npos) {
      const string sibling = dir.substr(0, parent) + "\\Blender Kernels";
      if (read_whole_file(sibling + "\\masks.txt", probe)) {
        return sibling;
      }
    }
  }
#  endif
  return "";
}

/* The iOS/iPadOS version Porthole is running on ("27.2"), which it writes for programs at every start. The GPU binaries of a pack only fit the version they were built for. */
string MetalDevice::ios_version()
{
  if (const char *env = getenv("CYCLES_WM_IOS_VERSION")) {
    return env;
  }
  vector<uint8_t> marker;
  if (read_whole_file("C:\\windows\\porthole-ios-version", marker)) {
    string v(marker.begin(), marker.end());
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) {
      v.pop_back();
    }
    return v;
  }
  return "";
}

string MetalDevice::gpu_architecture()
{
  if (gpu_arch_name.empty()) {
    char arch[32] = {0};
    obj_handle_t error = 0;
    MTLDevice_setBinaryArchive(mtlDevice, nullptr, 0, arch, &error);
    wm::error_text(error);
    gpu_arch_name = arch[0] ? string(arch) : string("unknown");
    fprintf(stderr, "[cycles-wm] GPU architecture %s\n", gpu_arch_name.c_str());
  }
  return gpu_arch_name;
}

/* Loads the kernels for exactly these features from the pack, if the pack has them for this GPU. */
bool MetalDevice::load_prebuilt_kernels(const uint features)
{
  const string dir = kernel_pack_dir();
  if (dir.empty() || getenv("CYCLES_WM_NO_PREBUILT")) {
    return false;
  }
  /* The pack that has every feature the scene needs, and the fewest others (a kernel with more features than the scene uses is what Cycles' own generic kernels are). */
  uint pack = 0;
  bool found = false;
  {
    vector<uint8_t> list;
    if (read_whole_file(dir + "/masks.txt", list)) {
      const string text(list.begin(), list.end());
      size_t pos = 0;
      while (pos < text.size()) {
        size_t end = text.find('\n', pos);
        if (end == string::npos) {
          end = text.size();
        }
        const uint mask = uint(strtoul(text.substr(pos, end - pos).c_str(), nullptr, 16));
        pos = end + 1;
        if (mask && (mask & features) == features) {
          if (!found || __builtin_popcount(mask) < __builtin_popcount(pack)) {
            pack = mask;
            found = true;
          }
        }
      }
    }
  }
  if (!found) {
    return false;
  }
  const string base = string_printf("%s/%08x", dir.c_str(), pack);
  vector<uint8_t> air, archive;
  if (!read_whole_file(base + ".metallib", air)) {
    return false;
  }
  const string arch = gpu_architecture();
  const string os = ios_version();
  prebuilt_library_only = false;
  const bool have_archive = !os.empty() && read_whole_file(base + "-" + arch + "-" + os + ".archive", archive);
  /* A pack marked "library-only-ok" (the universal pack) is used without GPU binaries: its device-independent shader code is loaded and the device only builds the GPU code from it. */
  vector<uint8_t> marker;
  const bool library_only_ok = read_whole_file(dir + "/library-only-ok", marker);
  if (!have_archive && !library_only_ok) {
    fprintf(stderr, "[cycles-wm] a kernel pack for features 0x%x exists, but not for this GPU and system (%s, iOS %s)\n", pack, arch.c_str(), os.c_str());
    return false;
  }
  const double started = time_dt();
  obj_handle_t error = 0;
  obj_handle_t data = DispatchData_alloc_init(uint64_t(uintptr_t(air.data())), air.size());
  obj_handle_t lib = data ? MTLDevice_newLibrary(mtlDevice, data, &error) : 0;
  const string lib_why = wm::error_text(error);
  wm::release(data);
  if (!lib) {
    fprintf(stderr, "[cycles-wm] the kernel pack's library would not load: %s\n", lib_why.substr(0, 160).c_str());
    return false;
  }
  if (!have_archive) {
    wm::release(mtlLibrary);
    mtlLibrary = lib;
    prebuilt_library_only = true;
    fprintf(stderr, "[cycles-wm] kernels for features 0x%x (the scene needs 0x%x) loaded from the pack in %.2f s (library %zu KB): the pack has no GPU binaries for %s on iOS %s, so the GPU code is built on this device from the pack's shader code\n",
            pack, features, time_dt() - started, air.size() >> 10, arch.c_str(), os.c_str());
    return true;
  }
  error = 0;
  const uint32_t ok = MTLDevice_setBinaryArchive(mtlDevice, archive.data(), archive.size(), nullptr, &error);
  const string why = wm::error_text(error);
  if (!ok) {
    fprintf(stderr, "[cycles-wm] the kernel pack's GPU binaries would not load: %s\n", why.substr(0, 160).c_str());
    wm::release(lib);
    return false;
  }
  wm::release(mtlLibrary);
  mtlLibrary = lib;
  prebuilt_active = true;
  fprintf(stderr, "[cycles-wm] kernels for features 0x%x (the scene needs 0x%x) loaded from the pack in %.2f s (library %zu KB, GPU binaries %zu KB): no compile on this device\n",
          pack, features, time_dt() - started, air.size() >> 10, archive.size() >> 10);
  return true;
}

/* Does this device take the pack's GPU binaries? The first kernel is built from them, with no compile; if that does not work the pack is dropped and the kernels are compiled as before. */
bool MetalDevice::probe_prebuilt()
{
  obj_handle_t fn = MTLLibrary_newFunction(mtlLibrary, "cycles_metal_integrator_reset");
  if (!fn) {
    return false;
  }
  WMTComputePipelineInfoEx info = {};
  info.compute_function = fn;
  info.max_total_threads_per_threadgroup = 64;
  info.flags = 1u | 2u;
  info.immutable_buffers = 0x7;
  obj_handle_t error = 0;
  obj_handle_t pso = MTLDevice_newComputePipelineStateEx(mtlDevice, &info, &error);
  const string why = wm::error_text(error);
  wm::release(fn);
  if (!pso) {
    fprintf(stderr, "[cycles-wm] the pack's GPU binaries do not fit this device (%s); compiling the kernels here instead\n", why.empty() ? "no matching pipeline in the archive" : why.substr(0, 160).c_str());
    return false;
  }
  wm::release(pso);
  return true;
}

bool MetalDevice::load_kernels(const uint _kernel_features)
{
  fprintf(stderr, "[cycles-wm] load_kernels requested 0x%x (had 0x%x)\n", _kernel_features, kernel_features);
  kernel_features |= _kernel_features;
  /* Test knob: extra feature bits (hex) forced on, to compile the kernels the way a scene with those features would. */
  if (const char *extra = getenv("CYCLES_WM_FEATURES_OR")) {
    kernel_features |= uint(strtoul(extra, nullptr, 0));
  }
  if (drop_denoising) {
    kernel_features &= ~uint(KERNEL_FEATURE_DENOISING);
  }
  /* Kernels built ahead of time for exactly these features: used as they are, with every feature the scene needs. */
  {
    const string key = string_printf("prebuilt:%08x", kernel_features);
    if (mtlLibrary && key == library_features_key) {
      return true;
    }
    join_precompile();
    prebuilt_active = false;
    bool from_pack = !prebuilt_disabled && load_prebuilt_kernels(kernel_features);
    if (from_pack && !prebuilt_library_only && !probe_prebuilt()) {
      from_pack = false;
      prebuilt_active = false;
      prebuilt_disabled = true;
      wm::release(mtlLibrary);
    }
    if (from_pack) {
      library_features_key = key;
      for (MetalPipeline &p : pipelines) {
        wm::release(p.pipeline);
        wm::release(p.function);
        p = MetalPipeline();
      }
      start_precompile();
      return true;
    }
  }
#  ifdef _WIN32
  /* Kernels built with transparent shadows kill the system's Metal compiler on the phone and the iPad (a scene that is otherwise the same,
   * without them, compiles fine), and after that no kernel can be built until the app is restarted. Under Porthole they are left out:
   * shadows of transparent surfaces then come out solid. CYCLES_WM_KEEP_TRANSPARENT=1 brings them back. */
  {
    const char *keep = getenv("CYCLES_WM_KEEP_TRANSPARENT");
    if ((kernel_features & KERNEL_FEATURE_TRANSPARENT) && !(keep && atoi(keep) != 0)) {
      fprintf(stderr, "[cycles-wm] leaving out transparent shadows (the system's compiler cannot build those kernels); shadows of transparent surfaces will be solid\n");
      kernel_features &= ~uint(KERNEL_FEATURE_TRANSPARENT);
    }
  }
#  endif
  return compile_library();
}

bool MetalDevice::is_ready(string &status) const
{
  if (!error_msg.empty()) {
    return true;
  }
  status = "";
  if (precompile_active.load() > 0) {
    status = string_printf("Compiling render kernels %d/%d (the first time takes a few minutes)",
                           precompile_done.load(),
                           precompile_total);
    return false;
  }
  return mtlLibrary != 0;
}

bool MetalDevice::build_pipeline(const DeviceKernel kernel, MetalPipeline &p, string &error_text)
{
  /* Test knob for the recovery path: this kernel "fails" while the denoising feature is on. */
  if (getenv("CYCLES_WM_TEST_FAIL") && kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND && (kernel_features & KERNEL_FEATURE_DENOISING)) {
    error_text = "test failure";
    return false;
  }

  const string function_name = string("cycles_metal_") + device_kernel_as_string(kernel);
  p.function = MTLLibrary_newFunction(mtlLibrary, function_name.c_str());
  if (!p.function) {
    error_text = string_printf("Could not find the Metal function \"%s\"", function_name.c_str());
    return false;
  }

  /* Cycles tunes the threadgroup size per GPU generation and kernel (a copy of the table in the
   * original Metal device). The sort passes REQUIRE 1024 threads per group: the kernel derives
   * its partition from the group index, and any other size writes outside the offsets buffer. */
  int threads_per_threadgroup = max_threads_per_threadgroup;
  int num_threads_per_block = 0;
  {
    struct Tuning {
      int threads_per_threadgroup;
      int num_threads_per_block;
    };
    Tuning t = {0, 0};
    const string name = wm::device_name(mtlDevice);
    const bool is_m1 = name.find("M1") != string::npos;
    const bool is_m2 = name.find("M2") != string::npos;
    const bool m2_big = is_m2 && (name.find("Pro") != string::npos ||
                                  name.find("Max") != string::npos ||
                                  name.find("Ultra") != string::npos);
    if (kernel == DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS ||
        kernel == DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS)
    {
      t = {1024, 1024};
    }
    else if (is_m1) {
      switch (kernel) {
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES: t = {256, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: t = {768, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: t = {512, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW: t = {384, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: t = {512, 64}; break;
        case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: t = {512, 256}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: t = {512, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW: t = {384, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: t = {576, 384}; break;
        case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: t = {832, 832}; break;
        default: break;
      }
    }
    else if (is_m2 && m2_big) {
      switch (kernel) {
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES: t = {384, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: t = {640, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: t = {1024, 64}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW: t = {704, 704}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: t = {640, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: t = {896, 768}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: t = {512, 128}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW: t = {32, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: t = {768, 576}; break;
        case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: t = {896, 768}; break;
        default: break;
      }
    }
    else if (is_m2) {
      switch (kernel) {
        case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES: t = {32, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: t = {832, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: t = {64, 64}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW: t = {64, 64}; break;
        case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: t = {704, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: t = {1024, 256}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: t = {64, 32}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW: t = {256, 256}; break;
        case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: t = {448, 384}; break;
        case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: t = {1024, 1024}; break;
        default: break;
      }
    }
    else {
      /* M3 and later (and every other GPU): peak occupancy through Dynamic Caching. */
      t = {64, 64};
    }
    if (t.threads_per_threadgroup) {
      threads_per_threadgroup = t.threads_per_threadgroup;
      num_threads_per_block = t.num_threads_per_block;
    }
  }

  WMTComputePipelineInfoEx info = {};
  info.compute_function = p.function;
  info.max_total_threads_per_threadgroup = threads_per_threadgroup;
  info.flags = 1u;              /* the threadgroup size is a multiple of the execution width */
  info.immutable_buffers = 0x7; /* buffers 0, 1 and 2 are never written by the kernels */
  if (prebuilt_active) {
    WMTComputePipelineInfoEx archive_info = info;
    archive_info.flags |= 2u;
    obj_handle_t archive_error = 0;
    p.pipeline = MTLDevice_newComputePipelineStateEx(mtlDevice, &archive_info, &archive_error);
    wm::error_text(archive_error);
    if (p.pipeline) {
      WMTComputePipelineLimits lim = {};
      MTLComputePipelineState_limits(p.pipeline, &lim);
      fprintf(stderr, "[cycles-wm] pipeline %s from the GPU binaries (no compile)\n", device_kernel_as_string(kernel));
      if (!num_threads_per_block) {
        num_threads_per_block = round_down(lim.max_total_threads_per_threadgroup, lim.thread_execution_width);
        num_threads_per_block = std::max(num_threads_per_block, (int)lim.thread_execution_width);
      }
      p.num_threads_per_block = num_threads_per_block;
      return true;
    }
  }
  const double starttime = time_dt();
  string text_error;
  /* The big shading kernels are compiled one at a time (three of them at once need more memory than a phone can spare next to a big scene),
   * and a compile the system interrupted (it stops the compiler service when memory runs short) is tried again after a pause. */
  const bool heavy = (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE || kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE ||
                      kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND || kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW ||
                      kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME || kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD ||
                      kernel == DEVICE_KERNEL_SHADER_EVAL_BACKGROUND);
  for (int attempt = 0; attempt < 2; attempt++) {
    obj_handle_t error = 0;
    {
      std::unique_lock<std::mutex> heavy_lock(heavy_compile_mutex, std::defer_lock);
      if (heavy) {
        heavy_lock.lock();
      }
      p.pipeline = MTLDevice_newComputePipelineStateEx(mtlDevice, &info, &error);
    }
    text_error = wm::error_text(error);
    if (p.pipeline || (text_error.find("INTERRUPTED") == string::npos && text_error.find("nterrupted") == string::npos)) {
      break;
    }
    size_t left_mb = 0;
#  ifdef _WIN32
    {
      MEMORYSTATUSEX status = {};
      status.dwLength = sizeof(status);
      if (GlobalMemoryStatusEx(&status)) {
        left_mb = size_t(status.ullAvailPhys >> 20);
      }
    }
#  endif
    fprintf(stderr, "[cycles-wm] building %s was interrupted by the system (memory left %zu MB); trying again in %d s (attempt %d)\n",
            device_kernel_as_string(kernel), left_mb, 2 * (attempt + 1), attempt + 2);
    std::this_thread::sleep_for(std::chrono::seconds(2 * (attempt + 1)));
  }
  if (!p.pipeline) {
    error_text = string_printf(
        "Failed to build the pipeline for %s:\n%s", device_kernel_as_string(kernel), text_error.c_str());
    return false;
  }
  {
    WMTComputePipelineLimits lim = {};
    MTLComputePipelineState_limits(p.pipeline, &lim);
    fprintf(stderr,
            "[cycles-wm] pipeline %s built in %.2f s (threads/group asked %d, pipeline allows %d, width %d)\n",
            device_kernel_as_string(kernel),
            time_dt() - starttime,
            threads_per_threadgroup,
            int(lim.max_total_threads_per_threadgroup),
            int(lim.thread_execution_width));
  }

  if (!num_threads_per_block) {
    WMTComputePipelineLimits limits = {};
    MTLComputePipelineState_limits(p.pipeline, &limits);
    num_threads_per_block = round_down(limits.max_total_threads_per_threadgroup,
                                       limits.thread_execution_width);
    num_threads_per_block = std::max(num_threads_per_block, (int)limits.thread_execution_width);
  }
  p.num_threads_per_block = num_threads_per_block;
  return true;
}

/* Builds one kernel from a library compiled for the given features, only to see whether the compiler gets through it. */
bool MetalDevice::try_variant(const uint features, const DeviceKernel kernel, string &why)
{
  string text = "\n#include \"kernel/device/metal/kernel.metal\"\n";
  text = path_source_replace_includes(text, path_get("source"));
  const string src = preprocess_source_for(features) + text;
  obj_handle_t error = 0;
  obj_handle_t lib = MTLDevice_newLibraryWithSource(mtlDevice, src.data(), src.size(), 0x0302, WMT_LIBRARY_FLAG_FAST_MATH, &error);
  string err = wm::error_text(error);
  if (!lib) {
    why = "library: " + err.substr(0, 120);
    return false;
  }
  const string name = string("cycles_metal_") + device_kernel_as_string(kernel);
  obj_handle_t fn = MTLLibrary_newFunction(lib, name.c_str());
  bool ok = false;
  if (fn) {
    WMTComputePipelineInfoEx info = {};
    info.compute_function = fn;
    info.max_total_threads_per_threadgroup = 64;
    info.flags = 1u;
    info.immutable_buffers = 0x7;
    error = 0;
    obj_handle_t pso = MTLDevice_newComputePipelineStateEx(mtlDevice, &info, &error);
    err = wm::error_text(error);
    ok = pso != 0;
    if (!ok) {
      why = err.substr(0, 120);
    }
    wm::release(pso);
    wm::release(fn);
  }
  else {
    why = "no such function";
  }
  wm::release(lib);
  return ok;
}

/* A kernel the compiler would not build: find out which scene feature is the problem (the same kernel without each of the suspects),
 * and if it is the denoising data, carry on without that feature instead of failing the render. */
void MetalDevice::recover_after_failure(const DeviceKernel kernel)
{
  const uint features = kernel_features;
  fprintf(stderr, "[cycles-wm] %s would not build with features 0x%x (%s); trying it without some of them\n",
          device_kernel_as_string(kernel), features, precompile_error.substr(0, 100).c_str());
  struct Suspect {
    const char *name;
    uint bits;
  };
  const Suspect suspects[] = {
      {"denoising", uint(KERNEL_FEATURE_DENOISING)},
      {"light tree", uint(KERNEL_FEATURE_LIGHT_TREE)},
      {"transparent", uint(KERNEL_FEATURE_TRANSPARENT)},
      {"emission nodes", uint(KERNEL_FEATURE_NODE_EMISSION)},
      {"denoising + light tree + transparent", uint(KERNEL_FEATURE_DENOISING | KERNEL_FEATURE_LIGHT_TREE | KERNEL_FEATURE_TRANSPARENT)},
  };
  bool denoising_fixes_it = false;
  for (const Suspect &s : suspects) {
    if (!(features & s.bits)) {
      continue;
    }
    string why;
    const double started = time_dt();
    const bool ok = try_variant(features & ~s.bits, kernel, why);
    fprintf(stderr, "[cycles-wm]   without %s: %s in %.1f s%s%s\n", s.name, ok ? "builds" : "FAILS", time_dt() - started, ok ? "" : " - ", why.c_str());
    if (ok && s.bits == uint(KERNEL_FEATURE_DENOISING)) {
      denoising_fixes_it = true;
    }
  }

  if (!denoising_fixes_it) {
    set_error(precompile_error.empty() ? "A render kernel could not be built." : precompile_error);
    return;
  }

  fprintf(stderr, "[cycles-wm] carrying on without the denoising data kernels (a denoiser then works without its guide passes)\n");
  drop_denoising = true;
  kernel_features &= ~uint(KERNEL_FEATURE_DENOISING);
  library_features_key = preprocess_source();
  string text = "\n#include \"kernel/device/metal/kernel.metal\"\n";
  text = path_source_replace_includes(text, path_get("source"));
  source = preprocess_source() + text;
  wm::release(mtlLibrary);
  obj_handle_t error = 0;
  mtlLibrary = MTLDevice_newLibraryWithSource(mtlDevice, source.data(), source.size(), 0x0302, WMT_LIBRARY_FLAG_FAST_MATH, &error);
  const string lib_error = wm::error_text(error);
  if (!mtlLibrary) {
    set_error("Failed to compile library:\n" + lib_error);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    for (MetalPipeline &p : pipelines) {
      wm::release(p.pipeline);
      wm::release(p.function);
      p = MetalPipeline();
    }
  }
  for (const DeviceKernel k : core_kernel_list()) {
    obj_handle_t probe = MTLLibrary_newFunction(mtlLibrary, (string("cycles_metal_") + device_kernel_as_string(k)).c_str());
    if (probe) {
      wm::release(probe);
      if (!get_pipeline(k, false)) {
        return; /* the error is set */
      }
    }
  }
}

const MetalPipeline *MetalDevice::get_pipeline(const DeviceKernel kernel, const bool quiet)
{
  MetalPipeline &p = pipelines[kernel];
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (p.pipeline) {
      return &p;
    }
    if (p.failed || !mtlLibrary) {
      return nullptr;
    }
  }

  /* Built outside the lock: the threads that build them up front work in parallel. */
  MetalPipeline built;
  string error_text;
  const bool ok = build_pipeline(kernel, built, error_text);

  {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (p.pipeline) { /* another thread was first */
      wm::release(built.pipeline);
      wm::release(built.function);
      return &p;
    }
    if (ok) {
      p = built;
      return &p;
    }
    p.failed = true;
  }
  wm::release(built.function);
  if (quiet) {
    /* The up-front builds report their failures themselves (recover_after_failure). */
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    precompile_error = error_text;
    return nullptr;
  }
  set_error(error_text);
  return nullptr;
}

/* ---- building the pipelines up front ---- */

/* Where the GPU memory goes: totals by kind, the biggest buffers, and the pixel copies the host still holds next to their textures. */
void MetalDevice::log_memory_breakdown(const char *when)
{
  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
  struct Item {
    size_t bytes;
    string name;
  };
  vector<Item> items;
  size_t buffers = 0, textures = 0, texture_host_copies = 0, host_backed = 0;
  for (auto &it : metal_mem_map) {
    buffers += it.second->size;
    items.push_back({size_t(it.second->size), it.second->mem ? it.second->mem->log_name() : string("?")});
  }
  for (auto &it : texture_map) {
    textures += it.second.size;
    device_memory *mem = it.first;
    if (mem && mem->host_pointer) {
      texture_host_copies += mem->memory_size();
    }
  }
  (void)host_backed;
  std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.bytes > b.bytes; });
  size_t left_mb = 0;
#  ifdef _WIN32
  {
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
      left_mb = size_t(status.ullAvailPhys >> 20);
    }
  }
#  endif
  fprintf(stderr, "[cycles-wm] memory (%s): %zu buffers %zu MB, %zu textures %zu MB, pixel copies the host still holds %zu MB, %zu MB left\n",
          when, metal_mem_map.size(), buffers >> 20, texture_map.size(), textures >> 20, texture_host_copies >> 20,
          left_mb);
  for (size_t i = 0; i < items.size() && i < 12; i++) {
    fprintf(stderr, "[cycles-wm]   %7.1f MB  %s\n", double(items[i].bytes) / 1048576.0, items[i].name.c_str());
  }
}

/* The path states (hundreds of MB) are allocated only after the kernels are built: the compiler service needs the room while it works. */
void MetalDevice::wait_for_kernels(const double max_seconds)
{
  const double started = time_dt();
  bool waited = false;
  while (precompile_active.load() > 0 && time_dt() - started < max_seconds) {
    waited = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (waited) {
    fprintf(stderr, "[cycles-wm] waited %.1f s for the kernels before allocating the path states\n", time_dt() - started);
  }
}

void MetalDevice::join_precompile()
{
  for (std::thread &t : precompile_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  precompile_threads.clear();
  precompile_active = 0;
}

/* The kernels nearly every render uses (a plain surface scene with lights). */
static const std::vector<DeviceKernel> &core_kernel_list()
{
  static const std::vector<DeviceKernel> list = {
      DEVICE_KERNEL_INTEGRATOR_RESET,
      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
      DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND,
      DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE,
      DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
      DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,
      DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS,
      DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS,
      DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES,
      DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY,
      DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES,
      DEVICE_KERNEL_SHADER_EVAL_BACKGROUND,
      DEVICE_KERNEL_PREFIX_SUM,
  };
  return list;
}

void MetalDevice::start_precompile()
{
  join_precompile();
  if (!mtlLibrary) {
    return;
  }

  /* The kernels nearly every render uses (a plain surface scene with lights). The rest (volumes, MNEE, raytrace
   * shading, subsurface, bake, denoising helpers) are built the first time a scene needs them: building all 77
   * took 244 s of compile time on an M4. */
  const std::vector<DeviceKernel> &core = core_kernel_list();
  precompile_next = 0;
  precompile_done = 0;
  precompile_failed_kernel = -1;
  precompile_total = int(core.size());
  int workers = 3;
#  ifdef _WIN32
  /* A phone next to a big scene: the system stops the compiler service when memory runs short, and every compile that runs
   * at the same time makes it bigger. With less than 3 GB left, one kernel is compiled at a time. */
  {
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
      fprintf(stderr, "[cycles-wm] kernel features 0x%x, library source %zu bytes, memory left %zu MB\n", kernel_features, source.size(), size_t(status.ullAvailPhys >> 20));
      if (status.ullAvailPhys < (3000ull << 20)) {
        workers = 1;
      }
    }
  }
#  endif
  if (const char *str = getenv("CYCLES_METAL_COMPILE_THREADS")) {
    workers = std::max(0, std::min(8, atoi(str)));
  }
  if (workers == 0) {
    return; /* every pipeline is then built the first time it is used */
  }
  precompile_active = workers;
  fprintf(stderr, "[cycles-wm] building %d core kernel pipelines on %d threads\n", precompile_total, workers);

  for (int w = 0; w < workers; w++) {
    precompile_threads.emplace_back([this, w, &core]() {
      (void)w;
      const double started = time_dt();
      for (;;) {
        const int index = precompile_next.fetch_add(1);
        if (index >= precompile_total) {
          break;
        }
        const DeviceKernel kernel = core[index];
        if (!have_error() && precompile_failed_kernel.load() < 0) {
          /* Kernels the library does not have are skipped here; the queue reports the ones it needs. */
          const string function_name = string("cycles_metal_") + device_kernel_as_string(kernel);
          obj_handle_t probe = MTLLibrary_newFunction(mtlLibrary, function_name.c_str());
          if (probe) {
            wm::release(probe);
            if (!get_pipeline(kernel, true)) {
              int none = -1;
              precompile_failed_kernel.compare_exchange_strong(none, int(kernel));
            }
          }
        }
        precompile_done++;
      }
      if (precompile_active.load() == 1 && precompile_failed_kernel.load() >= 0) {
        /* The last worker: nothing else is building. Try to find out why, and to carry on without it. */
        recover_after_failure(DeviceKernel(precompile_failed_kernel.load()));
      }
      if (--precompile_active == 0) {
        fprintf(stderr, "[cycles-wm] all kernel pipelines ready (%.1f s)\n", time_dt() - started);
      }
    });
  }
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

  if (total_mem_used() > recommended_working_set()) {
    fprintf(stderr,
            "[cycles-wm] out of GPU memory: %zu MB in use by all devices, budget %zu MB, last allocation %s (%zu MB)\n",
            total_mem_used() >> 20,
            recommended_working_set() >> 20,
            mem.log_name().c_str(),
            size >> 20);
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
  else if (mem.type == MEM_IMAGE_TEXTURE) {
    image_copy_to((device_image &)mem);
  }
  /* Other shared buffers: the host side already is the buffer. */
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

void MetalDevice::image_res_grow(const uint image_info_id)
{
  if (image_info_id >= image_info.size()) {
    /* Geometric growth to amortize reallocation cost. */
    const size_t new_size = max(size_t(image_info_id) + 128, image_info.size() * 2);
    image_info.resize(new_size);
  }
  if (image_info_id >= image_res.size()) {
    image_res.resize(max(size_t(image_info.size()), size_t(image_info_id) + 1));
  }
  const size_t needed = sizeof(uint64_t) * image_res.size();
  if (needed > image_bindings.size) {
    /* The bindings buffer grows with the image count; the queue refills it before each run. */
    metal_mem_free(image_bindings.handle, image_bindings.size);
    image_bindings = wm::new_buffer(mtlDevice, needed);
    metal_mem_alloc(image_bindings);
    if (image_bindings.contents) {
      memset(image_bindings.contents, 0, image_bindings.size);
    }
  }
}

void MetalDevice::image_alloc_as_buffer(device_image &mem)
{
  generic_alloc(mem);

  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
  const uint image_info_id = mem.image_info_id;
  image_res_grow(image_info_id);

  auto it = metal_mem_map.find(&mem);
  if (it == metal_mem_map.end()) {
    return;
  }
  ImageRes &res = image_res[image_info_id];
  res = ImageRes();
  res.handle = it->second->buffer.handle;
  res.gpu_address = it->second->buffer.gpu_address;
  res.is_buffer = true;
  image_info[image_info_id] = mem.info;
}

void MetalDevice::image_alloc(device_image &mem)
{
  /* Same limits as the original Metal device. 1D images are linear buffers. */
  if (mem.data_height > 0) {
    if (mem.data_width > 16384 || mem.data_height > 16384) {
      set_error(string_printf(
          "Texture exceeds maximum allowed size of 16384 x 16384 (requested: %zu x %zu)",
          mem.data_width,
          mem.data_height));
      return;
    }
  }
  else {
    image_alloc_as_buffer(mem);
    return;
  }

  const size_t size = mem.memory_size();

  /* sampler_index maps into the kernel's constant 'metal_samplers' array. */
  uint64_t sampler_index = mem.info.extension;
  if (mem.info.interpolation != INTERPOLATION_CLOSEST) {
    sampler_index += 4;
  }

  /* data_elements is 1, 2 or 4 (index 2 = three elements is not used). */
  static const WMTPixelFormat f_uchar[] = {
      WMTPixelFormatR8Unorm, WMTPixelFormatRG8Unorm, WMTPixelFormatInvalid, WMTPixelFormatRGBA8Unorm};
  static const WMTPixelFormat f_uint16[] = {WMTPixelFormatR16Unorm,
                                            WMTPixelFormatRG16Unorm,
                                            WMTPixelFormatInvalid,
                                            WMTPixelFormatRGBA16Unorm};
  static const WMTPixelFormat f_uint[] = {WMTPixelFormatR32Uint,
                                          WMTPixelFormatRG32Uint,
                                          WMTPixelFormatInvalid,
                                          WMTPixelFormatRGBA32Uint};
  static const WMTPixelFormat f_int[] = {WMTPixelFormatR32Sint,
                                         WMTPixelFormatRG32Sint,
                                         WMTPixelFormatInvalid,
                                         WMTPixelFormatRGBA32Sint};
  static const WMTPixelFormat f_float[] = {WMTPixelFormatR32Float,
                                           WMTPixelFormatRG32Float,
                                           WMTPixelFormatInvalid,
                                           WMTPixelFormatRGBA32Float};
  static const WMTPixelFormat f_half[] = {WMTPixelFormatR16Float,
                                          WMTPixelFormatRG16Float,
                                          WMTPixelFormatInvalid,
                                          WMTPixelFormatRGBA16Float};

  WMTPixelFormat format = WMTPixelFormatInvalid;
  const int e = mem.data_elements - 1;
  if (e >= 0 && e < 4) {
    switch (mem.data_type) {
      case TYPE_UCHAR:
        format = f_uchar[e];
        break;
      case TYPE_UINT16:
        format = f_uint16[e];
        break;
      case TYPE_UINT:
        format = f_uint[e];
        break;
      case TYPE_INT:
        format = f_int[e];
        break;
      case TYPE_FLOAT:
        format = f_float[e];
        break;
      case TYPE_HALF:
        format = f_half[e];
        break;
      default:
        break;
    }
  }
  if (format == WMTPixelFormatInvalid) {
    set_error("Unsupported image format");
    return;
  }

  WMTTextureInfo ti = {};
  ti.pixel_format = format;
  ti.width = (uint32_t)mem.data_width;
  ti.height = (uint32_t)mem.data_height;
  ti.depth = 1;
  ti.array_length = 1;
  ti.type = WMTTextureType2D;
  ti.mipmap_level_count = 1;
  ti.sample_count = 1;
  ti.usage = WMTTextureUsageShaderRead;
  ti.options = WMTResourceStorageModeShared;

  LOG_DEBUG << "Texture 2D allocate: " << mem.log_name() << ", "
            << string_human_readable_number(size) << " bytes.";

  obj_handle_t texture = MTLDevice_newTexture(mtlDevice, &ti);
  if (!texture) {
    set_error("System is out of GPU memory");
    return;
  }

  const size_t src_pitch = mem.data_width * datatype_size(mem.data_type) * mem.data_elements;
  WMTOrigin origin = {0, 0, 0};
  WMTSize extent = {mem.data_width, mem.data_height, 1};
  WMTMemoryPointer data;
  data.ptr = mem.host_pointer;
  MTLTexture_replaceRegion(texture, origin, extent, 0, 0, data, src_pitch, 0);

  mem.device_pointer = (device_ptr)texture;
  mem.device_size = size;
  metal_mem_alloc(texture, size);

  std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
  texture_map[&mem] = {texture, size};

  const uint image_info_id = mem.image_info_id;
  image_res_grow(image_info_id);

  ImageRes &res = image_res[image_info_id];
  res = ImageRes();
  res.handle = texture;
  res.gpu_id = ti.gpu_resource_id;
  image_info[image_info_id] = mem.info;
  image_info[image_info_id].data = uint64_t(image_info_id) | (sampler_index << 32);

  if (total_mem_used() > recommended_working_set()) {
    set_error("System is out of GPU memory");
  }
}

void MetalDevice::image_copy_to(device_image &mem)
{
  if (mem.is_resident(this) && mem.data_height > 0) {
    obj_handle_t texture = 0;
    {
      std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
      auto it = texture_map.find(&mem);
      if (it == texture_map.end()) {
        return;
      }
      texture = it->second.handle;
    }
    const size_t src_pitch = mem.data_width * datatype_size(mem.data_type) * mem.data_elements;
    WMTOrigin origin = {0, 0, 0};
    WMTSize extent = {mem.data_width, mem.data_height, 1};
    WMTMemoryPointer data;
    data.ptr = mem.host_pointer;
    MTLTexture_replaceRegion(texture, origin, extent, 0, 0, data, src_pitch, 0);
  }
}

void MetalDevice::image_free(device_image &mem)
{
  const uint image_info_id = mem.image_info_id;
  if (mem.data_height == 0) {
    if (mem.device_pointer) {
      generic_free(mem);
    }
  }
  else {
    std::lock_guard<std::recursive_mutex> lock(metal_mem_map_mutex);
    auto it = texture_map.find(&mem);
    if (it != texture_map.end()) {
      metal_mem_free(it->second.handle, it->second.size);
      texture_map.erase(it);
      mem.device_pointer = 0;
      mem.device_size = 0;
    }
  }
  if (image_info_id < image_res.size()) {
    image_res[image_info_id] = ImageRes();
  }
}

void MetalDevice::load_image_info()
{
  /* Ensure image_info is allocated before populating. */
  image_info.copy_to_device();

  uint64_t *bindings = (uint64_t *)image_bindings.contents;
  if (!bindings) {
    return;
  }
  memset(bindings, 0, image_bindings.size);
  const size_t count = min(size_t(image_info.size()), image_res.size());
  for (size_t id = 0; id < count; ++id) {
    const ImageRes &res = image_res[id];
    if (!res.handle) {
      continue;
    }
    if (res.is_buffer) {
      /* The GPU address of a 1D buffer image goes into the image info itself. */
      image_info[id].data = res.gpu_address;
    }
    else {
      bindings[id] = res.gpu_id;
    }
  }
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
