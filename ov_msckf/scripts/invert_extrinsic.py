#!/usr/bin/env python3
"""Convert a camera-to-IMU 4x4 transform into OpenVINS T_cam_imu."""

import sys


def determinant3(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inverse3(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = determinant3(matrix)
    if abs(determinant) < 1e-12:
        raise ValueError("旋转矩阵不可逆")

    cofactors = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    return [[value / determinant for value in row] for row in cofactors]


def project_to_rotation(matrix):
    if determinant3(matrix) <= 0:
        raise ValueError("旋转矩阵行列式必须为正数")

    rotation = [row[:] for row in matrix]
    for _ in range(50):
        inverse_transpose = list(map(list, zip(*inverse3(rotation))))
        updated = [
            [(rotation[r][c] + inverse_transpose[r][c]) * 0.5 for c in range(3)]
            for r in range(3)
        ]
        difference = max(
            abs(updated[r][c] - rotation[r][c]) for r in range(3) for c in range(3)
        )
        rotation = updated
        if difference < 1e-12:
            break

    correction = max(
        abs(rotation[r][c] - matrix[r][c]) for r in range(3) for c in range(3)
    )
    if correction > 0.05:
        raise ValueError("输入的左上 3x3 矩阵不像旋转矩阵，请检查输入")
    return rotation


def parse_matrix(lines):
    matrix = []
    for line in lines:
        if ":" in line:
            line = line.split(":", 1)[1]
        values = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        try:
            row = [float(value) for value in values]
        except ValueError as error:
            raise ValueError("矩阵行必须只包含数字、逗号或方括号") from error
        if len(row) != 4:
            raise ValueError("矩阵每行必须恰好包含 4 个数字")
        matrix.append(row)

    if len(matrix) != 4:
        raise ValueError("必须输入恰好 4 行矩阵")
    if any(abs(matrix[3][index] - value) > 1e-6
           for index, value in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("齐次矩阵最后一行必须是 [0, 0, 0, 1]")
    return matrix


def invert_camera_to_imu(matrix):
    rotation = project_to_rotation([row[:3] for row in matrix[:3]])
    rotation_transpose = list(map(list, zip(*rotation)))
    translation = [matrix[row][3] for row in range(3)]
    inverse_translation = [
        -sum(rotation_transpose[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [
        rotation_transpose[row] + [inverse_translation[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def main():
    print("逐行输入 T_C0toI（相机到 IMU），共 4 行；逗号可保留：")
    try:
        matrix = parse_matrix([input("  ") for _ in range(4)])
        result = invert_camera_to_imu(matrix)
    except (EOFError, ValueError) as error:
        print("错误：{}".format(error), file=sys.stderr)
        return 2

    print("T_cam_imu:")
    for row in result:
        print("  - [{}]".format(", ".join("{:.8f}".format(value) for value in row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())