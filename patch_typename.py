# Patch calculationbase.h to add `typename` before dependent template iterators
# GCC ≥ 4.7 requires `typename` for dependent names; MSVC never did.
# This file contains 4 instances of `vector<...>::iterator` in template contexts
# that need `typename` prepended.

import re, sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

original = content

# Pattern 1: `vector<T>::iterator` → `typename vector<T>::iterator`
content = re.sub(
    r'for \(vector<T>::iterator',
    'for (typename vector<T>::iterator',
    content
)

# Pattern 2: `vector<NetExpectedDataSource<T>*>::iterator`
content = re.sub(
    r'for \(vector<NetExpectedDataSource<T>\*>::iterator',
    'for (typename vector<NetExpectedDataSource<T>*>::iterator',
    content
)

# Pattern 3: non-template but explicit type: `vector<NetExpectedDataSource<float>*>::iterator`
content = re.sub(
    r'for \(vector<NetExpectedDataSource<float>\*>::iterator',
    'for (typename vector<NetExpectedDataSource<float>*>::iterator',
    content
)

# Pattern 4: `vector<NetExpectedDataSource<string>*>::iterator`
content = re.sub(
    r'for \(vector<NetExpectedDataSource<string>\*>::iterator',
    'for (typename vector<NetExpectedDataSource<string>*>::iterator',
    content
)

# Also fix const_iterator patterns
content = re.sub(
    r'for \(vector<NetExpectedDataSource<T>\*>::const_iterator',
    'for (typename vector<NetExpectedDataSource<T>*>::const_iterator',
    content
)

changes = (content != original)
with open(path, 'w') as f:
    f.write(content)

print(f"{'Patched' if changes else 'No changes'} — {path}")
sys.exit(0 if changes else 0)
