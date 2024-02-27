import os
import pytest

from feflow.utils import lambdaprotocol

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'


def test_lambda_protocol():
    """

    Tests LambdaProtocol, ensures that it can be instantiated with defaults, and that it fails if disallowed functions are tried

    """

    # check that it's possible to instantiate a LambdaProtocol for all the default types
    for protocol in ['default', 'namd', 'quarters']:
        lp = lambdaprotocol.LambdaProtocol(functions=protocol)

    # check that if we give an incomplete set of parameters it will add in the missing terms
    missing_functions = {'lambda_sterics_delete': lambda x : x}
    lp = lambdaprotocol.LambdaProtocol(functions=missing_functions)
    assert (len(missing_functions) == 1)
    assert(len(lp.get_functions()) == 9)


def test_lambda_protocol_failure_ends():
    bad_function = {'lambda_sterics_delete': lambda x : -x}
    with pytest.raises(AssertionError):
        lp = lambdaprotocol.LambdaProtocol(functions=bad_function)


def test_lambda_protocol_naked_charges():
    naked_charge_functions = {'lambda_sterics_insert':
                  lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                  'lambda_electrostatics_insert':
                  lambda x: 2.0 * x if x < 0.5 else 1.0}
    with pytest.raises(AssertionError):
        lp = lambdaprotocol.LambdaProtocol(functions=naked_charge_functions)