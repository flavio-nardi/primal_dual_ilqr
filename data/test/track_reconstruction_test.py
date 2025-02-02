import os

import numpy as np

from data.read_track_data import read_raceline_data, read_track_data, resample
from data.track_reconstruction import read_track_from_json, track_reconstruction

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    x, y, w_right, w_left = read_track_data("Suzuka", curr_dir + "/../")
    x_raceline, y_raceline = read_raceline_data(
        "Suzuka_raceline", curr_dir + "/../"
    )
    track_waypoints = np.vstack((x, y))
    track_boundaries = np.vstack((w_left, w_right))
    raceline = np.vstack((x_raceline, y_raceline))
    ds = 1.0
    resampled_track_waypoints, _ = resample(
        track_waypoints, track_boundaries, raceline, ds
    )

    path = track_reconstruction(resampled_track_waypoints[:2, :], ds)

    assert (
        path.x[-1] == track_waypoints[0, -1]
        and path.y[-1] == track_waypoints[1, -1]
    )

    centerline_path, _ = read_track_from_json("Suzuka")

    assert (
        centerline_path.x[0] == path.x[0] and centerline_path.y[0] == path.y[0]
    )
    assert (
        centerline_path.x[-1] == path.x[-1]
        and centerline_path.y[-1] == path.y[-1]
    )
