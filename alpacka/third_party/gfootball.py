"""Source code copied and adjusted from gfootball.

GitHub:
https://github.com/google-research/football

License:
Copyright 2019 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np


def _get_cv2():
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        return cv2
    except ImportError:
        raise ImportError(
            'Could not import cv2. Install GFootball and its dependencies '
            'according to https://github.com/google-research/football.'
        )


class _TextWriter:

    def __init__(
        self, frame, x, y=0, field_coords=False, color=(255, 255, 255)
    ):
        self._frame = frame
        if field_coords:
            x = 400 * (x + 1) - 5
            y = 695 * (y + 0.43)
        self._pos_x = int(x)
        self._pos_y = int(y) + 20
        self._color = color

    def write(self, text, scale_factor=1):
        cv2 = _get_cv2()
        font = cv2.FONT_HERSHEY_SIMPLEX  # pylint: disable=no-member
        text_pos = (self._pos_x, self._pos_y)
        font_scale = 0.5 * scale_factor
        line_type = 1
        cv2.putText(self._frame, text, text_pos, font, font_scale, self._color,  # pylint: disable=no-member
                    line_type)
        self._pos_y += int(20 * scale_factor)


def get_frame(trace):
    """Visualizes a state based on a GFootball trace dict."""
    cv2 = _get_cv2()
    frame = np.uint8(np.zeros((600, 800, 3)))
    corner1 = (0, 0)
    corner2 = (799, 0)
    corner3 = (799, 599)
    corner4 = (0, 599)
    line_color = (0, 255, 255)
    cv2.line(frame, corner1, corner2, line_color)  # pylint: disable=no-member
    cv2.line(frame, corner2, corner3, line_color)  # pylint: disable=no-member
    cv2.line(frame, corner3, corner4, line_color)  # pylint: disable=no-member
    cv2.line(frame, corner4, corner1, line_color)  # pylint: disable=no-member
    cv2.line(frame, (399, 0), (399, 799), line_color)  # pylint: disable=no-member
    writer = _TextWriter(
        frame,
        trace['ball'][0],
        trace['ball'][1],
        field_coords=True,
        color=(255, 0, 0))
    writer.write('B')
    for player_idx, player_coord in enumerate(trace['left_team']):
        writer = _TextWriter(
            frame,
            player_coord[0],
            player_coord[1],
            field_coords=True,
            color=(0, 255, 0))
        letter = 'H'
        if 'left_agent_controlled_player' in trace and player_idx in trace[
            'left_agent_controlled_player']:
            letter = 'X'
        writer.write(letter)
    for player_idx, player_coord in enumerate(trace['right_team']):
        writer = _TextWriter(
            frame,
            player_coord[0],
            player_coord[1],
            field_coords=True,
            color=(255, 255, 0))
        letter = 'A'
        if (
            'opponent_active' in trace and
            player_idx in trace['opponent_active']
        ):
            letter = 'Y'
        elif 'right_agent_controlled_player' in trace and player_idx in trace[
            'right_agent_controlled_player']:
            letter = 'Y'
        writer.write(letter)
    return frame
