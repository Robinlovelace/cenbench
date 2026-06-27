## Summary

**Goal:** Build `sdna_vs2008.so` with OpenMP multithreading support on Linux, enabling parallel execution on multi-core machines.

**Approach:** Direct `g++` compilation bypassing the MSVC-targeted CMake build system (which is tightly coupled to Visual Studio generators, vcpkg, and Windows-only assumptions). The pre-built PyPI wheel ships a `.so` compiled without OpenMP, meaning sDNA runs single-threaded even on multi-core machines.

**Target platform:** Ubuntu 24.04, GCC 10/13, Boost 1.83 (system-installed), MuParser bundled in-tree.

---

## Compilation Command

```
g++-10 -std=c++14 -O2 -fPIC -fopenmp -fpermissive \
  -I sdna-plus/sDNA/sdna_vs2008 \
  -I sdna-plus/sDNA/muparser/drop/include \
  -c sdna-plus/sDNA/sdna_vs2008/<file>.cpp \
  -o build/<file>.o
```

Linked with: `g++ ... -shared -lgomp -ldl -lpthread -o sdna_vs2008.so`

---

## Error 1: CMakeLists.txt — MSVC/Ninja incompatibility

**File:** `sDNA/sdna_vs2008/CMakeLists.txt` (lines 37–52, 83–96, 498–500)

**Problem:** The CMakeLists.txt is written exclusively for Visual Studio generators:
- `cmake_minimum_required(VERSION 3.26)` — fine, but then it uses `CMAKE_VS_PLATFORM_NAME` (line 37) which is only set by Visual Studio generators, not by Ninja or Unix Makefiles.
- Compiler/linker flags are gated behind `if(MSVC)` blocks, with no Linux/GCC equivalent.
- Lines 498–500 unconditionally define `WIN32`, `_WINDOWS`, `_USRDLL` as global compile definitions, which pollutes the GCC preprocessor namespace.

**Fix:** On Linux, bypass CMake entirely and compile directly with `g++`. Or, if using CMake, pass `-G Ninja` and set platform explicitly, then either guard `CMAKE_VS_PLATFORM_NAME` usage behind `if(MSVC)` or provide a Linux generator path.

---

## Error 2: `targetver.h` — Windows-only header

**File:** `sDNA/sdna_vs2008/targetver.h`, line 8

**Error:**
```
fatal error: SDKDDKVer.h: No such file or directory
    8 | #include <SDKDDKVer.h>
```

**Problem:** `<SDKDDKVer.h>` is a Windows SDK header. It does not exist on Linux.

**Fix:** Wrap in `#ifdef _WINDOWS` / `#ifdef _WIN32`:
```cpp
#ifdef _WIN32
#include <SDKDDKVer.h>
#endif
```

---

## Error 3: `stdafx.h` — Windows-only includes + backslash paths + deprecated Boost

**File:** `sDNA/sdna_vs2008/stdafx.h`

### Error 3a: `<windows.h>` and `<wininet.h>` (lines 21–22)

```
fatal error: windows.h: No such file or directory
   21 | #include <windows.h>
fatal error: wininet.h: No such file or directory
   22 | #include <wininet.h>
```

**Fix:** Wrap in `#ifdef _WIN32`:
```cpp
#ifdef _WIN32
#include <windows.h>
#include <wininet.h>
#endif
```

### Error 3b: Backslash include path (line 26)

```
fatal error: IteratorTypeErasure\any_iterator\any_iterator.hpp: No such file or directory
   26 | #include "IteratorTypeErasure\any_iterator\any_iterator.hpp"
```

**Fix:** Use forward slashes (GCC on Linux treats `\` as a literal filename character):
```cpp
#include "IteratorTypeErasure/any_iterator/any_iterator.hpp"
```

### Error 3c: Deprecated Boost header (line 63)

```
warning: boost/geometry/multi/geometries/multi_point.hpp: This include file is deprecated and will be removed in Boost 1.86
   63 | #include <boost/geometry/multi/geometries/multi_point.hpp>
```

**Problem:** `boost/geometry/multi/` headers are deprecated since Boost 1.75 and removed in Boost ≥1.86. On Ubuntu 24.04 with Boost 1.83, this produces a warning that becomes a hard error on newer Boost.

**Fix:** Replace with the modern equivalent:
```cpp
#include <boost/geometry/geometries/multi_point.hpp>
```
(See also existing issue #33.)

---

## Error 4: `calculationbase.h` — missing `typename` in dependent template iterator

**File:** `sDNA/sdna_vs2008/calculationbase.h`, lines 112, 130, 141–142

**Error (GCC 10+):**
```
error: need 'typename' before 'std::vector<T>::iterator' because 'std::vector<T>' is a dependent scope
  112 |   for (vector<T>::iterator it=v.begin();it!=v.end();it++)
```

Similar errors on lines 130 and 141–142 for `vector<NetExpectedDataSource<T>*>::iterator` and `vector<NetExpectedDataSource<T>*>::const_iterator`.

**Problem:** GCC enforces the C++ standard rule that dependent template types must be prefixed with `typename`. MSVC is more permissive.

**Fix:** Add `typename`:
```cpp
for (typename vector<T>::iterator it=v.begin(); it!=v.end(); ++it)
```

Line 130 becomes:
```cpp
for (typename vector<NetExpectedDataSource<T>*>::iterator it=source.begin(); it!=source.end(); ++it)
```

Line 142 becomes:
```cpp
for (typename vector<NetExpectedDataSource<T>*>::const_iterator it=simplified_data.begin(); it!=simplified_data.end(); ++it)
```

(See also existing issue #59 — filed for Zig/Clang, but the main branch still has this unfixed.)

---

## Error 5: `calc_output_code.cpp` — temporaries bound to non-const references via OUTPUT macro

**File:** `sDNA/sdna_vs2008/calc_output_code.cpp`, line 5, and all call sites (lines 12–149)

**Error (GCC 10+):**
```
error: cannot bind non-const lvalue reference of type 'OutputDataWrapper&' to an rvalue of type 'OutputDataWrapper'
    5 | #define OUTPUT(x) output_map.add_output(ExtraNameWrapper((x),output_name_prefix,output_name_postfix))
      |                                            ^~~~~~~~~~~~~~
```

**Root cause:** The `OUTPUT(x)` macro passes expressions like `SDNAPolylineConnectivityOutputDataWrapper()` (a temporary/rvalue) to `ExtraNameWrapper`, whose constructor takes `OutputDataWrapper &od` (non-const lvalue reference). GCC disallows this; MSVC accepts it as an extension.

The `OUTPUT` macro expands to:
```cpp
output_map.add_output(ExtraNameWrapper((x),output_name_prefix,output_name_postfix))
```

Which in turn calls `OutputMap::add_output(OutputDataWrapper &output)` (line 519 of `sdna_output_utils.h`) — also a non-const lvalue reference.

Call sites producing temporaries include lines:
- 12: `OUTPUT(SDNAPolylineConnectivityOutputDataWrapper());`
- 13: `OUTPUT(SDNAPolylineLengthOutputDataWrapper());`
- 14: `OUTPUT(PolylineIndexedArrayOutputDataWrapper("Link Fraction","LFrac",&link_fraction));`
- 15: `OUTPUT(SDNAPolylineAngularCostOutputDataWrapper());`
- ...and ~40+ more on lines 16–149.

**Fix:** Change `OutputMap::add_output` and `ExtraNameWrapper::ExtraNameWrapper` (and downstream `ControlledRadialOutputDataWrapper` constructors) to take `const` references:

```cpp
// sdna_output_utils.h line 519
void add_output(const OutputDataWrapper &output)

// sdna_output_utils.h line 404
ExtraNameWrapper(const OutputDataWrapper &od, string pre, string post)

// sdna_output_utils.h lines 123, 137
ControlledRadialOutputDataWrapper(const RadialOutputDataWrapper &data, ...)
```

Alternatively, use the approach from the Zig/Clang fork (issue #53): replace the macro with class methods and use temporary variables, or make the `clone()` method `const` on the wrapper classes.

---

## Error 6: `sdna_output_utils.h` — non-const references blocking temporaries

**File:** `sDNA/sdna_vs2008/sdna_output_utils.h`

All the same non-const reference problems as Error 5, but declared in the header:

- **Line 404:** `ExtraNameWrapper(OutputDataWrapper &od, ...)` — takes non-const ref
- **Lines 123, 132, 137:** `ControlledRadialOutputDataWrapper(RadialOutputDataWrapper &data, ...)` — takes non-const ref
- **Line 519:** `void add_output(OutputDataWrapper &output)` — takes non-const ref

**Fix:** Change parameter types to `const OutputDataWrapper &` / `const RadialOutputDataWrapper &` and make `clone()` a `const` method on all wrapper classes.

---

## Pre-built wheel note

The PyPI wheel (`pip install sdna-plus`) works on Linux out of the box, but the bundled `sdna_vs2008.so` is compiled **without** OpenMP. Running `readelf -d sdna_vs2008.so` shows no `NEEDED libgomp` entry, and `nm -D` shows zero `GOMP_*` symbols. This means sDNA runs single-threaded even on multi-core machines.

To verify:
```bash
ldd $(python3 -c "import sdna_plus, os; print(os.path.join(os.path.dirname(sdna_plus.__file__),'x64','sdna_vs2008.so'))") | grep gomp
```
returns nothing.

---

## References

- Existing issue #33: Boost `multi_point.hpp` deprecation warning
- Existing issue #48: `std::auto_ptr` in MuParser for C++17+ compilers
- Existing issue #53: OUTPUT macro replaced with class functions (Zig/Clang fork)
- Existing issue #55: C++14 language standard set in CMake
- Existing issue #59: `typename` added to for-loop variables (Zig/Clang fork)
- Existing issue #60: Assignment vs update class function (Zig/Clang fork)
- Cross_platform branch: https://github.com/fiftysevendegreesofrad/sdna_plus/tree/Cross_platform
