"""
Pytest suite for PCA (pca.py).

"""

import numpy as np
import pytest

from vectora_ml.decomposition.pca import PCA
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def correlated_data():
    """
    Data with an obvious dominant direction: y is roughly 2*x plus a
    small amount of noise in a second, near-orthogonal direction.
    Useful for sanity-checking which direction PCA picks as PC1.
    """
    rng = np.random.RandomState(42)
    x = rng.normal(0, 10, size=200)
    y = 2 * x + rng.normal(0, 0.5, size=200)
    return np.column_stack([x, y])


@pytest.fixture
def simple_3d_data():
    rng = np.random.RandomState(0)
    return rng.normal(size=(50, 3))


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_n_components_is_none(self):
        pca = PCA()
        assert pca.n_components is None

    def test_accepts_positive_int(self):
        pca = PCA(n_components=2)
        assert pca.n_components == 2

    @pytest.mark.parametrize("bad_value", [0, -1, -5])
    def test_rejects_non_positive(self, bad_value):
        with pytest.raises(InvalidParameterError):
            PCA(n_components=bad_value)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, simple_3d_data):
        pca = PCA()
        result = pca.fit(simple_3d_data)
        assert result is pca

    def test_fit_sets_expected_attributes(self, simple_3d_data):
        pca = PCA(n_components=2).fit(simple_3d_data)
        assert hasattr(pca, "mean_")
        assert hasattr(pca, "components_")
        assert hasattr(pca, "explained_variance_")
        assert hasattr(pca, "explained_variance_ratio_")
        assert pca.n_features_in_ == 3

    def test_components_shape(self, simple_3d_data):
        pca = PCA(n_components=2).fit(simple_3d_data)
        assert pca.components_.shape == (2, 3)

    def test_full_components_shape_when_none(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        assert pca.components_.shape == (3, 3)

    def test_mean_is_column_mean(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        np.testing.assert_allclose(pca.mean_, simple_3d_data.mean(axis=0))

    def test_eigenvalues_sorted_descending(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        ev = pca.explained_variance_
        assert np.all(ev[:-1] >= ev[1:])

    def test_explained_variance_ratio_sums_to_one_when_all_kept(
        self, simple_3d_data
    ):
        pca = PCA().fit(simple_3d_data)
        assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)

    def test_explained_variance_ratio_sums_to_less_when_truncated(
        self, simple_3d_data
    ):
        pca = PCA(n_components=1).fit(simple_3d_data)
        assert pca.explained_variance_ratio_.sum() <= 1.0

    def test_n_components_exceeding_features_raises(self, simple_3d_data):
        pca = PCA(n_components=10)
        with pytest.raises(InvalidParameterError):
            pca.fit(simple_3d_data)

    def test_dominant_direction_recovered(self, correlated_data):
        # y ~= 2x means the first principal component direction should
        # be closely aligned with (1, 2) normalized, up to overall sign.
        pca = PCA(n_components=1).fit(correlated_data)
        pc1 = pca.components_[0]
        expected_direction = np.array([1, 2]) / np.linalg.norm([1, 2])
        # PCA components are only defined up to sign, so compare the
        # absolute value of the cosine similarity.
        cosine_similarity = np.abs(np.dot(pc1, expected_direction))
        assert cosine_similarity > 0.99

    def test_components_are_orthonormal(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)

    def test_constant_feature_zero_variance(self):
        rng = np.random.RandomState(1)
        X = np.column_stack(
            [rng.normal(size=50), np.full(50, 7.0)]
        )
        pca = PCA().fit(X)
        # One eigenvalue should be (near) zero since one feature never
        # varies.
        assert np.min(pca.explained_variance_) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------

class TestTransform:
    def test_transform_output_shape(self, simple_3d_data):
        pca = PCA(n_components=2).fit(simple_3d_data)
        transformed = pca.transform(simple_3d_data)
        assert transformed.shape == (50, 2)

    def test_transform_before_fit_raises(self, simple_3d_data):
        pca = PCA()
        with pytest.raises(Exception):
            pca.transform(simple_3d_data)

    def test_transform_dimension_mismatch_raises(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        wrong_shape = simple_3d_data[:, :2]
        with pytest.raises(DimensionMismatchError):
            pca.transform(wrong_shape)

    def test_transformed_data_is_centered(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        transformed = pca.transform(simple_3d_data)
        np.testing.assert_allclose(
            transformed.mean(axis=0), np.zeros(3), atol=1e-8
        )

    def test_transformed_components_uncorrelated(self, simple_3d_data):
        pca = PCA().fit(simple_3d_data)
        transformed = pca.transform(simple_3d_data)
        cov = np.cov(transformed, rowvar=False)
        off_diagonal = cov - np.diag(np.diagonal(cov))
        np.testing.assert_allclose(off_diagonal, 0, atol=1e-8)


# ---------------------------------------------------------------------------
# fit_transform()
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_matches_separate_fit_and_transform(self, simple_3d_data):
        pca_a = PCA(n_components=2)
        combined = pca_a.fit_transform(simple_3d_data)

        pca_b = PCA(n_components=2)
        pca_b.fit(simple_3d_data)
        separate = pca_b.transform(simple_3d_data)

        np.testing.assert_allclose(combined, separate)


# ---------------------------------------------------------------------------
# inverse_transform()
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_perfect_reconstruction_when_all_components_kept(
        self, simple_3d_data
    ):
        pca = PCA().fit(simple_3d_data)
        transformed = pca.transform(simple_3d_data)
        reconstructed = pca.inverse_transform(transformed)
        np.testing.assert_allclose(reconstructed, simple_3d_data, atol=1e-8)

    def test_lossy_reconstruction_when_truncated(self, correlated_data):
        pca = PCA(n_components=1).fit(correlated_data)
        transformed = pca.transform(correlated_data)
        reconstructed = pca.inverse_transform(transformed)
        # Reconstruction error should be small but not exactly zero,
        # since one dimension of information was dropped.
        error = np.linalg.norm(reconstructed - correlated_data)
        assert error > 0

    def test_inverse_transform_before_fit_raises(self):
        pca = PCA()
        with pytest.raises(Exception):
            pca.inverse_transform(np.zeros((5, 2)))