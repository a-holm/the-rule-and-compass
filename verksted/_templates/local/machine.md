# Machine profile

## Profile keys

One line per key, `- KEY: value`. A script that reads a key exits 2 and names it while the value is `<set me>`. Meanings: `_shared/local.md`.

- GATES: <set me>
- MACHINE_SCRIPTS: <set me>
- BUILD_VOLUME: <set me>
- DISK_MIN_FREE: <set me>
- BUILD_SLOT: <set me>
- BUILD_SLOTS: <set me>
- TEST_LOCK: <set me>
- PIN_CORES_BUILD: <set me>
- CORES_RESERVED: <set me>
- CAMPAIGN_BUILD_CAP: <set me>
- BUILD_SLOTS_BUSY: <set me>
- PIN_CORES_BUSY: <set me>
- MEM_MIN_FREE: <set me>
- BUILD_JOBS: <set me>
- BUILD_SLOT_WAIT_MAX: <set me>
- SANITIZER: <set me>
- CUDA_ARCH: <set me>
- ELAN_BIN: <set me>

## Hardware

- CPU model: <model>
- Physical cores: <n>
- Hyperthread pairs (from `lscpu -e`): <core pairs>
- RAM: <size>
- Disks: <device, size, mount, use>

## GPU

- Model: <model>
- Memory: <size>
- Driver: <version>
- CUDA toolkit: <version>

## Shared use

- Desktop priority: <rule>
- Reserved cores: <list>

## Throughput

| Date | Workload | Shape | Flags | Mask | Result | Record |
|---|---|---|---|---|---|---|
