load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "track_reconstruction_test",
    srcs = ["track_reconstruction_test.py"],
    deps = [
        "//data:read_track_data",
        "//data:track_reconstruction",
        "@pypi_numpy//:pkg",
    ],
)
