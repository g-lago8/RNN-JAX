def test_import_rnn_jax():
    import importlib

    rnn = importlib.import_module("rnn_jax")
    assert hasattr(rnn, "__version__")
    assert isinstance(rnn.__version__, str)
