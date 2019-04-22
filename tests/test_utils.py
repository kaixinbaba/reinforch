from reinforch.utils import read_config, tensor2tensor, obj2tensor, \
    FloatTensor


def test_read_config():
    s = read_config('tests/configs/test_network_config.json')
    assert s is not None


def test_tensor2tensor():
    t = FloatTensor([1, 2, 3, 4])
    feed = tensor2tensor(t, feed_network=True)
    assert isinstance(feed, FloatTensor)
    assert feed.size()[0] == 1 and feed.size()[1] == 4
    feed = tensor2tensor(t, target_shape=[2, 2])
    assert feed.size()[0] == 2 and feed.size()[1] == 2


def test_obj2tensor():
    o = [1, 2, 3, 4]
    feed = obj2tensor(o, feed_network=True)
    assert isinstance(feed, FloatTensor)
    assert feed.size()[0] == 1 and feed.size()[1] == 4
    feed = obj2tensor(o, target_shape=[2, 2])
    assert feed.size()[0] == 2 and feed.size()[1] == 2
