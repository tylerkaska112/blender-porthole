/* SPDX-FileCopyrightText: 2021-2026 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Cycles' Metal device written against the winemetal C interface (DXMT's Metal layer for Windows programs, with the Porthole additions: patches 0183-0186) instead of
 * Objective-C. The same sources run natively on macOS (winemetal built with DXMT_NATIVE) and inside Porthole on iOS (Windows build of Cycles calling winemetal.dll). */

#pragma once

#ifdef WITH_METAL

#  include <string>

#  include "winemetal.h"

#  include "util/string.h"
#  include "util/vector.h"

#  ifdef _WIN32
#    include "util/windows.h"
#    include <psapi.h>
#  endif

CCL_NAMESPACE_BEGIN

namespace wm {

inline string string_of(const obj_handle_t nsstring)
{
  if (!nsstring) {
    return "";
  }
  const uint64_t n = NSString_lengthOfBytesUsingEncoding(nsstring, WMTUTF8StringEncoding);
  string out(n + 1, '\0');
  NSString_getCString(nsstring, &out[0], n + 1, WMTUTF8StringEncoding);
  out.resize(strlen(out.c_str()));
  return out;
}

/* An autorelease pool for the Metal calls of one scope. Calls like commandBuffer, computeCommandEncoder and description hand back objects that are only autoreleased: without a pool
 * on the calling thread they stay alive until the thread ends (a render runs on a thread of its own, and a long one piles up thousands of command buffers). What has to outlive the
 * scope must be retained inside it (NSObject_retain) and released by whoever keeps it. */
struct AutoreleasePool {
  obj_handle_t pool;
  AutoreleasePool() : pool(NSAutoreleasePool_alloc_init()) {}
  ~AutoreleasePool()
  {
    if (pool) {
      NSObject_release(pool);
    }
  }
  AutoreleasePool(const AutoreleasePool &) = delete;
  AutoreleasePool &operator=(const AutoreleasePool &) = delete;
};

/* What the process holds, in MB, for the log lines that show where memory stays after a render: Metal's allocations on this device and (on Windows) the private bytes of the program. */
inline size_t metal_allocated_mb(const obj_handle_t device)
{
  return size_t(MTLDevice_currentAllocatedSize(device) >> 20);
}

inline size_t process_private_mb()
{
#  ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX counters = {};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&counters, sizeof(counters))) {
    return size_t(counters.PrivateUsage >> 20);
  }
#  endif
  return 0;
}

/* The text of an NSError handle (and releases it). */
inline string error_text(obj_handle_t error)
{
  if (!error) {
    return "";
  }
  AutoreleasePool pool;
  const string text = string_of(NSObject_description(error));
  NSObject_release(error);
  return text;
}

inline void release(obj_handle_t &handle)
{
  if (handle) {
    NSObject_release(handle);
    handle = 0;
  }
}

/* The Metal devices of this machine, in the order Cycles numbers them. The array stays alive for the life of the process. */
const vector<obj_handle_t> &devices();
inline string device_name(const obj_handle_t device)
{
  AutoreleasePool pool;
  return string_of(MTLDevice_name(device));
}

struct Buffer {
  obj_handle_t handle = 0;
  void *contents = nullptr; /* null for private buffers */
  uint64_t gpu_address = 0;
  size_t size = 0;
};

inline Buffer new_buffer(const obj_handle_t device, const size_t size, const bool shared = true)
{
  Buffer b;
  WMTBufferInfo info = {};
  info.length = size;
  info.options = shared ? WMTResourceStorageModeShared : WMTResourceStorageModePrivate;
  info.memory.ptr = nullptr;
  b.handle = MTLDevice_newBuffer(device, &info);
  if (b.handle) {
    b.contents = info.memory.ptr;
    b.gpu_address = info.gpu_address;
    b.size = size;
  }
  return b;
}

} /* namespace wm */

CCL_NAMESPACE_END

#endif /* WITH_METAL */
