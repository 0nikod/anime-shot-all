import gradio as gr

from anime_shot_all.gui import build_app


def test_gui_builds_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)
