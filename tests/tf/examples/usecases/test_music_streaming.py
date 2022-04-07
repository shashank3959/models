import pytest
from testbook import testbook


@testbook("examples/usecases/music-streaming.ipynb", execute=True)
@pytest.mark.example
def test_tf_example_music_streaming_usecase(tb):
    assert tb.cell_output_text(2)[-19:] == "parameters: 1094466"
    assert tb.cell_output_text(4)[-19:] == "parameters: 2006533"