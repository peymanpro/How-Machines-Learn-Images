def test_environment_is_ready() -> None:
    import numpy
    import PIL

    assert numpy.__version__ == "2.5.3"
    assert PIL.__version__ == "12.3.0"
