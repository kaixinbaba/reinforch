def test_read_config():
    from reinforch.utils import read_config
    s = read_config('tests/configs/test_network_config.json')
    assert s is not None
