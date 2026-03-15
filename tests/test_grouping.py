from src.core.grouping import get_module_key


def test_root_level_file_is_file_key() -> None:
    key, kind = get_module_key("main.cpp", depth=1)
    assert key == "main.cpp"
    assert kind == "file"


def test_exactly_two_levels_is_grouped_to_top_folder() -> None:
    key, kind = get_module_key("drivers/foo.cpp", depth=1)
    assert key == "drivers"
    assert kind == "folder"


def test_deeper_path_groups_by_depth_1() -> None:
    key, kind = get_module_key("drivers/imu/foo.cpp", depth=1)
    assert key == "drivers"
    assert kind == "folder"


def test_deeper_path_groups_by_depth_2() -> None:
    key, kind = get_module_key("drivers/imu/foo.cpp", depth=2)
    assert key == "drivers/imu"
    assert kind == "folder"


def test_windows_separator_is_normalized() -> None:
    key, kind = get_module_key(r"drivers\imu\foo.cpp", depth=2)
    assert key == "drivers/imu"
    assert kind == "folder"
