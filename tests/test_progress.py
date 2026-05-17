from anime_shot_all.progress import normalize_process_output


def test_normalize_process_output_splits_carriage_returns_and_trims():
    text = "[matroska,webm @ 0000026fdf687940] Unsupported encoding type\rframe=1\r\n"

    assert normalize_process_output(text) == "[matroska,webm @ 0000026fdf687940] Unsupported encoding type\nframe=1"


def test_normalize_process_output_handles_missing_trailing_newline():
    text = "first line\nsecond line"

    assert normalize_process_output(text) == "first line\nsecond line"
