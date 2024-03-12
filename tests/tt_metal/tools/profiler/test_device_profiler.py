# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os, sys
import json
import re
import inspect
import pytest

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    clear_profiler_runtime_artifacts,
)

PROG_EXMP_DIR = "programming_examples/profiler"


def run_device_profiler_test(doubleRun=False, setup=False):
    name = inspect.stack()[1].function
    clear_profiler_runtime_artifacts()
    profilerRun = os.system(f"cd {TT_METAL_HOME} && " f"build/{PROG_EXMP_DIR}/{name}")
    assert profilerRun == 0

    if doubleRun:
        # Run test under twice to make sure icache is populated
        # with instructions for test
        profilerRun = os.system(f"cd {TT_METAL_HOME} && " f"build/{PROG_EXMP_DIR}/{name}")
        assert profilerRun == 0

    setupStr = ""
    if setup:
        setupStr = f"-s {name}"

    postProcessRun = os.system(
        f"cd {PROFILER_SCRIPTS_ROOT} && " f"./process_device_log.py {setupStr} --no-artifacts --no-print-stats"
    )

    assert postProcessRun == 0, f"Log process script crashed with exit code {postProcessRun}"

    devicesData = {}
    with open(f"{PROFILER_ARTIFACTS_DIR}/output/device/device_analysis_data.json", "r") as devicesDataJson:
        devicesData = json.load(devicesDataJson)

    return devicesData


def get_function_name():
    frame = inspect.currentframe()
    return frame.f_code.co_name


def test_multi_op():
    OP_COUNT = 1100
    RUN_COUNT = 2
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RUN_COUNT, 88 * OP_COUNT * RUN_COUNT],
        "wormhole_b0": [72 * OP_COUNT * RUN_COUNT, 64 * OP_COUNT * RUN_COUNT, 56 * OP_COUNT * RUN_COUNT],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    statName = f"BRISC KERNEL_START->KERNEL_END"

    assert statName in stats.keys(), "Wrong device analysis format"
    assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


def test_custom_cycle_count():
    REF_CYCLE_COUNT_PER_LOOP = 52
    LOOP_COUNT = 2000
    REF_CYCLE_COUNT = REF_CYCLE_COUNT_PER_LOOP * LOOP_COUNT
    REF_CYCLE_COUNT_HIGH_MULTIPLIER = 10
    REF_CYCLE_COUNT_LOW_MULTIPLIER = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT * REF_CYCLE_COUNT_HIGH_MULTIPLIER
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT // REF_CYCLE_COUNT_LOW_MULTIPLIER

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert stats[statName]["stats"]["Average"] < REF_CYCLE_COUNT_MAX, "Wrong cycle count, too high"
        assert stats[statName]["stats"]["Average"] > REF_CYCLE_COUNT_MIN, "Wrong cycle count, too low"


def test_full_buffer():
    OP_COUNT = 25
    RISC_COUNT = 5
    ZONE_COUNT = 125
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RISC_COUNT * ZONE_COUNT, 88 * OP_COUNT * RISC_COUNT * ZONE_COUNT],
        "wormhole_b0": [
            72 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            64 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            56 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
        ],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]
    statName = "Marker Repeat"

    assert statName in stats.keys(), "Wrong device analysis format"
    assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"
