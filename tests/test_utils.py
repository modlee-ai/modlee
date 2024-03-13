import pytest
from modlee.utils import discretize

def test_discretize():

    n = 0.234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.00234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 2.34
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 30143215
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = [3.3, 32144321, 0.032]
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = (1, 23)
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = "test"
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.0005985885113477707
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))


