import numpy as np
import pytest
from demo.tracking_evidence import extra_lateral_extent


def test_no_error_and_translation():
    assert extra_lateral_extent(0,0,9.4,9.4,3.74)==0
    np.testing.assert_allclose(extra_lateral_extent(np.array([.3,-.3]),np.zeros(2),9.4,9.4,3.74),[.3,.3])


def test_rotation_and_length_sensitivity():
    h=np.deg2rad(2)
    expected=9.4*np.sin(h)+1.87*np.cos(h)-1.87
    assert float(extra_lateral_extent(0,h,9.4,9.4,3.74))==pytest.approx(expected)
    assert extra_lateral_extent(0,h,9.4,9.4,3.74)>extra_lateral_extent(0,h,3,3,3.74)
    assert extra_lateral_extent(0,-h,9.4,9.4,3.74)==pytest.approx(expected)


def test_invalid_dimensions_and_missing_measurement():
    with pytest.raises(ValueError): extra_lateral_extent(0,0,-1,2,3)
    with pytest.raises(ValueError): extra_lateral_extent(np.nan,0,1,2,3)
