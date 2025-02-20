import pytest


class TestDummy:
    @pytest.fixture(scope='class', autouse=True)
    def setup(self):
        print("setup_class called")
        pytest.test = "hallo"

    def test_1(self):
        print(pytest.test)
        assert True

    def test_2(self):
        print(pytest.test)
        assert True