load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "read_track_data",
    srcs = ["read_track_data.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pypi_matplotlib//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_binary(
    name = "track_reconstruction",
    srcs = ["track_reconstruction.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":read_track_data",
        "@pypi_casadi//:pkg",
        "@pypi_matplotlib//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_scipy//:pkg",
    ],
)
