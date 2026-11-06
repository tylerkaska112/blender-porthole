/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 * SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "device/device.h"

#ifdef WITH_METAL

#  include <set>

#  include "device/metal_wm/device.h"
#  include "device/metal_wm/device_impl.h"
#  include "device/metal_wm/wm.h"

#  include "util/log.h"
#  include "util/string.h"

CCL_NAMESPACE_BEGIN

namespace wm {

const vector<obj_handle_t> &devices()
{
  static vector<obj_handle_t> all;
  static bool done = false;
  if (!done) {
    done = true;
    const obj_handle_t array = WMTCopyAllDevices(); /* kept for the life of the process: the devices belong to it */
    const uint64_t n = array ? NSArray_count(array) : 0;
    for (uint64_t i = 0; i < n; i++) {
      all.push_back(NSArray_object(array, i));
    }
  }
  return all;
}

} /* namespace wm */

unique_ptr<Device> device_metal_create(const DeviceInfo &info,
                                       Stats &stats,
                                       Profiler &profiler,
                                       bool headless)
{
  return make_unique<MetalDevice>(info, stats, profiler, headless);
}

bool device_metal_init()
{
  return !wm::devices().empty();
}

void device_metal_exit() {}

void device_metal_info(vector<DeviceInfo> &devices)
{
  std::set<string> unique_ids;
  int device_index = 0;
  for (const obj_handle_t device : wm::devices()) {
    const string device_name = wm::device_name(device);
    string id = string("METAL_") + device_name;
    if (unique_ids.count(id)) {
      id += string_printf("_ID_%d", device_index);
    }
    unique_ids.insert(id);

    DeviceInfo info;
    info.type = DEVICE_METAL;
    info.description = string_remove_trademark(device_name);
    info.num = device_index;
    info.display_device = true;
    info.denoisers = DENOISER_NONE;
    info.id = id;
    info.has_nanovdb = false;
    info.has_mnee_ = true;
    /* Hardware ray tracing is not in this port yet (Cycles' own BVH is used). */
    info.use_hardware_raytracing = false;
    info.use_metalrt_by_default = false;

    devices.push_back(info);
    device_index++;
    LOG_INFO << "Added device \"" << info.description << "\" with id \"" << info.id << "\".";
  }
}

string device_metal_capabilities()
{
  string result;
  const vector<obj_handle_t> &all = wm::devices();
  if (all.empty()) {
    return "No Metal devices found\n";
  }
  result += string_printf("Number of devices: %u\n", (uint32_t)all.size());
  for (const obj_handle_t device : all) {
    result += string_printf("\t\tDevice: %s\n", wm::device_name(device).c_str());
  }
  return result;
}

CCL_NAMESPACE_END

#else

CCL_NAMESPACE_BEGIN
CCL_NAMESPACE_END

#endif /* WITH_METAL */
