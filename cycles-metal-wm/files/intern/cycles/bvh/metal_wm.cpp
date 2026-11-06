/* SPDX-FileCopyrightText: 2026 The Porthole authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* The Metal device written against winemetal has no hardware ray tracing yet and asks for Cycles' own BVH, so there is no Metal acceleration structure to create. */

#ifdef WITH_METAL

#  include "bvh/metal.h"

CCL_NAMESPACE_BEGIN

unique_ptr<BVH> bvh_metal_create(const BVHParams & /*params*/,
                                 const vector<Geometry *> & /*geometry*/,
                                 const vector<Object *> & /*objects*/,
                                 Device * /*device*/)
{
  return nullptr;
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
