#!/usr/bin/env python3
"""Generates the winemetal layer that the Cycles Metal device (Blender) is built against.

  make_win_layer.py <winemetal source dir> <names file> <out dir>

Reads winemetal.h, winemetal_thunks.h and winemetal_thunks.c (DXMT, LGPL-2.1-or-later, with Porthole's
patches) and writes, for just the functions listed in <names file>:
  <out>/winemetal.h       the declarations those functions need (enums, structs, prototypes)
  <out>/winemetal_win.cpp their implementations for a Windows program: every call goes straight to the
                          unix side of winemetal through Wine's unix-call dispatcher, with the slot
                          numbers and parameter structs the engine itself uses. winemetal.dll is only
                          loaded (that is what makes Wine load winemetal's unix library); none of its
                          exports are needed, so it works with a Porthole whose winemetal.dll was
                          not rebuilt.
The same winemetal.h works for the macOS test build (-DDXMT_NATIVE=1 and the native library).
"""
import re
import sys
from pathlib import Path

src, names_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
wanted = [l.strip() for l in names_file.read_text().split() if l.strip()]
roots_extra = [w for w in wanted if not re.match(r"^(MTL|NS|WMT)[A-Za-z0-9]*(_[A-Za-z0-9]+)?$", w)]


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    return t


DEFINES = {}


def preprocess(text, defined):
    """Evaluates #if/#ifdef/#else/#endif (simple forms) and drops every other directive."""
    lines = text.split("\n")
    out, stack = [], []  # stack of (active, taken, parent_active)

    def active():
        return all(s[0] for s in stack)

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("#"):
            # join continuations
            full = s
            while full.endswith("\\") and i + 1 < len(lines):
                i += 1
                full = full[:-1] + " " + lines[i].strip()
            d = re.match(r"#\s*(\w+)\s*(.*)", full)
            kw, arg = d.group(1), d.group(2).strip()
            if kw in ("ifdef", "ifndef"):
                v = (arg.split()[0] in defined) == (kw == "ifdef")
                stack.append([v, v])
            elif kw == "if":
                e = arg
                e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in defined else "0", e)
                e = re.sub(r"defined\s+(\w+)", lambda m: "1" if m.group(1) in defined else "0", e)
                e = e.replace("||", " or ").replace("&&", " and ").replace("!", " not ")
                e = re.sub(r"\b[A-Za-z_]\w*\b(?<!and)(?<!or)(?<!not)", lambda m: m.group(0) if m.group(0) in ("and", "or", "not") else "0", e)
                v = bool(eval(e))
                stack.append([v, v])
            elif kw == "else":
                top = stack[-1]
                top[0] = not top[1]
                top[1] = top[1] or top[0]
            elif kw == "endif":
                stack.pop()
            elif kw == "define" and active():
                dm = re.match(r"(\w+)(?:\([^)]*\))?\s+(.*)", arg)
                if dm and "(" not in arg.split()[0]:
                    DEFINES.setdefault(dm.group(1), "#define %s %s" % (dm.group(1), dm.group(2)))
            # other directives (include, pragma) are dropped
        elif active():
            out.append(line)
        i += 1
    return "\n".join(out)


def split_items(text):
    """Top-level declarations: ended by ';' at brace/paren depth 0."""
    items, cur, depth = [], [], 0
    for ch in text:
        cur.append(ch)
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        elif ch == ";" and depth == 0:
            items.append("".join(cur).strip())
            cur = []
    rest = "".join(cur).strip()
    if rest:
        items.append(rest)
    return [i for i in items if i]


def defined_names(item):
    m = re.match(r"enum\s+(\w+)", item)
    if m:
        return [m.group(1)]
    m = re.match(r"struct\s+(\w+)\s*\{", item)
    if m:
        return [m.group(1)]
    m = re.match(r"STATIC_ASSERT", item)
    if m:
        return []
    m = re.match(r"typedef\b.*?(\w+)\s*;$", item, re.S)
    if m:
        return [m.group(1)]
    m = re.match(r"WINEMETAL_API\b.*?(\w+)\s*\(", item, re.S)
    if m:
        return [m.group(1)]
    return []


def enum_values(item):
    if not item.startswith("enum"):
        return []
    body = item[item.index("{") + 1:item.rindex("}")]
    return re.findall(r"\b(\w+)\s*(?:=|,|$)", body)


defs = {}  # name -> item index
header_items, thunk_items = [], []

h = preprocess(strip_comments((src / "winemetal.h").read_text()), {"__cplusplus"})
for it in split_items(h):
    header_items.append(it)
t = preprocess(strip_comments((src / "winemetal_thunks.h").read_text()), {"__cplusplus"})
for it in split_items(t):
    thunk_items.append(it)

all_items = [("h", i) for i in header_items] + [("t", i) for i in thunk_items]
name_to_item = {}
asserts = {}
for idx, (_, it) in enumerate(all_items):
    for n in defined_names(it):
        name_to_item.setdefault(n, idx)
    for v in enum_values(it):
        name_to_item.setdefault(v, idx)
    m = re.match(r"STATIC_ASSERT\s*\(\s*sizeof\s*\(\s*(?:struct\s+)?(\w+)\s*\)", it)
    if m:
        asserts.setdefault(m.group(1), []).append(idx)

# thunk function bodies
thunks_c = strip_comments((src / "winemetal_thunks.c").read_text())
funcs = {}
for m in re.finditer(r"WINEMETAL_API\s+([^\n(]+?)\s*\n?\s*(\w+)\s*\(", thunks_c):
    name = m.group(2)
    start = m.start()
    i = thunks_c.index("{", m.end())
    depth = 0
    j = i
    while True:
        if thunks_c[j] == "{":
            depth += 1
        elif thunks_c[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = thunks_c[start:j + 1]
    rtype = m.group(1).strip()
    if rtype != "void":
        # the thunks are C: make the (implicit) conversion of the result explicit for C++
        body = re.sub(r"return (params\.\w+);", lambda r: "return (%s)%s;" % (rtype, r.group(1)), body)
    funcs.setdefault(name, body)

missing = [w for w in wanted if w not in funcs and w not in name_to_item and w not in DEFINES]
if missing:
    sys.exit("not found: " + " ".join(missing))

# closure of what the wanted functions need
needed_h = set()
needed_t = set()
work = []
ident = re.compile(r"\b[A-Za-z_]\w*\b")
for w in wanted:
    if w in funcs:
        # prototype from the header
        if w in name_to_item:
            work.append(name_to_item[w])
        for n in ident.findall(funcs[w]):
            if n in name_to_item:
                work.append(name_to_item[n])
    elif w in name_to_item:
        work.append(name_to_item[w])
seen = set()
while work:
    idx = work.pop()
    if idx in seen:
        continue
    seen.add(idx)
    kind, it = all_items[idx]
    for n in ident.findall(it):
        if n in name_to_item and name_to_item[n] != idx:
            work.append(name_to_item[n])
    for n in defined_names(it):
        for a in asserts.get(n, []):
            work.append(a)
for idx in sorted(seen):
    (needed_h if all_items[idx][0] == "h" else needed_t).add(idx)

BANNER = """/* Subset of winemetal's interface - generated by tools/spikes/cycles_metal/make_win_layer.py.
 * winemetal is part of DXMT, Copyright (c) 2023-2026 Feifan He for CodeWeavers, with the changes
 * of Wine's Madeira and Porthole; licensed under the GNU Lesser General Public License,
 * version 2.1 or (at your option) any later version (see COPYING.LIB). */
"""


def fmt(it):
    it = re.sub(r"[ \t]+", " ", it)
    it = re.sub(r"\n\s*\n+", "\n", it)
    return it + "\n"


hdr = [BANNER, "#ifndef _WINEMETAL_H\n#define _WINEMETAL_H\n\n#include <cstdint>\n#include <cassert>\n",
       "#define STATIC_ASSERT(x) static_assert(x)\n",
       '#ifndef WINEMETAL_API\n#define WINEMETAL_API extern "C"\n#endif\n',
       "typedef uint64_t obj_handle_t;\n#define NULL_OBJECT_HANDLE 0\n",
       "#ifndef _MACH_PORT_T\ntypedef uint32_t mach_port_t;\n#endif\n"]
for idx in sorted(needed_h):
    it = all_items[idx][1]
    if re.match(r"typedef\s+uint64_t\s+obj_handle_t|typedef\s+uint32_t\s+mach_port_t", it):
        continue
    hdr.append(fmt(it))
used = set(ident.findall("\n".join(all_items[i][1] for i in needed_h)))
for w in wanted:
    used.add(w)
for n, line in DEFINES.items():
    if n in used and n not in ("STATIC_ASSERT", "WINEMETAL_API", "NULL_OBJECT_HANDLE", "WMT_MEMPTR_SET"):
        hdr.append(line + "\n")
hdr.append("#endif\n")
header_text = "\n".join(hdr)
# the WMT_MEMPTR_SET macro of the real header (plain pointers: no 32-bit guests here)
header_text = header_text.replace("#endif\n\n", "#endif\n\n", 1)

cpp = [BANNER, "#include <windows.h>\n#include <cstdint>\n#include <cstring>\n",
       '#include "winemetal.h"\n',
       "#define WMT_MEMPTR_SET(obj, value) (obj).ptr = (value)\n",
       "#define PtrToUInt64(v) ((uint64_t)(uintptr_t)(v))\n",
       "#pragma pack(push, 8)\n"]
for idx in sorted(needed_t):
    cpp.append(fmt(all_items[idx][1]))
cpp.append("#pragma pack(pop)\n")
cpp.append(r'''
/* The unix side of winemetal. Loading winemetal.dll is what makes Wine load winemetal's unix library;
 * its unix-call table is then reached through ntdll's dispatcher (what the DLL's own thunks do). */
namespace {
typedef LONG (*UnixCallFn)(UINT64 handle, unsigned int code, void *args);
typedef LONG (NTAPI *QueryVirtualMemoryFn)(HANDLE, PVOID, ULONG, PVOID, SIZE_T, SIZE_T *);

struct UnixLayer {
  UINT64 handle = 0;
  UnixCallFn call = nullptr;
  UnixLayer() {
    HMODULE metal = LoadLibraryA("winemetal.dll");
    HMODULE nt = GetModuleHandleA("ntdll.dll");
    if (!metal || !nt) {
      return;
    }
    QueryVirtualMemoryFn query = (QueryVirtualMemoryFn)GetProcAddress(nt, "NtQueryVirtualMemory");
    UnixCallFn *dispatcher = (UnixCallFn *)GetProcAddress(nt, "__wine_unix_call_dispatcher");
    if (!query || !dispatcher || !*dispatcher) {
      return;
    }
    UINT64 h = 0;
    /* 1000 = MemoryWineLoadUnixLib */
    if (query(GetCurrentProcess(), metal, 1000, &h, sizeof h, nullptr) != 0) {
      return;
    }
    handle = h;
    call = *dispatcher;
  }
};

LONG wm_unix_call(unsigned int code, void *params) {
  static UnixLayer layer;
  return layer.call ? layer.call(layer.handle, code, params) : -1;
}
} // namespace

#define UNIX_CALL(code, params) wm_unix_call(code, params)
#define WINEMETAL_API extern "C"
''')
for w in wanted:
    if w in funcs:
        cpp.append(funcs[w] + "\n")
cpp_text = "\n".join(cpp)
# the thunks use the census hook only through UNIX_CALL, which is replaced above

out.mkdir(parents=True, exist_ok=True)
(out / "winemetal.h").write_text(header_text)
(out / "winemetal_win.cpp").write_text(cpp_text)
print("functions:", len([w for w in wanted if w in funcs]), "header items:", len(needed_h), "thunk structs:", len(needed_t))
