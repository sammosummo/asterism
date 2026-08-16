"""The public numerical relationship-matrix builders."""

from __future__ import annotations

import numpy as np
import pytest

import asterism


class _ArrayHookList(list):
    def __init__(self, visible, converted):
        super().__init__(visible)
        self._converted = converted

    def __array__(self, dtype=None, copy=None):
        result = self._converted
        if dtype is not None:
            result = result.astype(dtype)
        return result.copy() if copy else result


class _DataHookArray(np.ndarray):
    def __new__(cls, visible, replacement):
        result = np.asarray(visible).view(cls)
        result._replacement = replacement
        return result

    @property
    def _data(self):
        return self._replacement


class _DtypeHookArray(np.ndarray):
    def __new__(cls, values, reported_dtype):
        result = np.asarray(values).view(cls)
        result._reported_dtype = np.dtype(reported_dtype)
        return result

    @property
    def dtype(self):
        return self._reported_dtype


class _StatefulInt(int):
    def __new__(cls, value, changed_value):
        result = int.__new__(cls, value)
        result._changed_value = changed_value
        result._float_calls = 0
        return result

    def __float__(self):
        self._float_calls += 1
        if self._float_calls < 4:
            return float(int(self))
        return float(self._changed_value)


class _StatefulNumpyInt(np.int64):
    __module__ = "numpy"

    def __new__(cls, value, changed_value):
        result = np.int64.__new__(cls, value)
        result._changed_value = changed_value
        result._float_calls = 0
        return result

    def __float__(self):
        self._float_calls += 1
        if self._float_calls < 4:
            return float(np.int64(self))
        return float(self._changed_value)


class _FloatHookZeroDimensionalArray(np.ndarray):
    def __new__(cls, visible_value, converted_value):
        result = np.asarray(visible_value).reshape(()).view(cls)
        result._converted_value = converted_value
        return result

    def __float__(self):
        return float(self._converted_value)


def _object_array(shape, *values):
    result = np.empty(shape, dtype=object)
    result.flat[:] = values
    return result


def test_weighted_linear_gene_matrix_follows_the_equation():
    genotypes = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    built = asterism.gene_linear_matrix(
        genotypes,
        variant_weights=np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(
        built,
        np.array([[4.0, 8.0, 0.0], [8.0, 17.0, 2.0], [0.0, 2.0, 4.0]]),
    )
    assert isinstance(built, np.ndarray)


def test_burden_gene_matrix_is_the_rank_one_weighted_burden():
    genotypes = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    built = asterism.gene_burden_matrix(
        genotypes,
        variant_weights=np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(
        built,
        np.array([[4.0, 10.0, 4.0], [10.0, 25.0, 10.0], [4.0, 10.0, 4.0]]),
    )
    assert np.linalg.matrix_rank(built) == 1


def test_gene_constructions_are_positive_semidefinite():
    matrices = [
        asterism.gene_linear_matrix(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
        ),
        asterism.gene_burden_matrix(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
        ),
    ]

    for matrix in matrices:
        assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_linear_gene_matrix_is_invariant_to_reordering_variants_with_weights():
    genotypes = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    original = asterism.gene_linear_matrix(
        genotypes, variant_weights=np.array([1.0, 2.0])
    )
    reordered = asterism.gene_linear_matrix(
        genotypes[:, ::-1], variant_weights=np.array([2.0, 1.0])
    )

    np.testing.assert_array_equal(original, reordered)


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[0.0, np.nan], [1.0, 2.0]])
            ),
            "GENE_MATRIX_GENOTYPE_NOT_FINITE",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[0.0, 3.0], [1.0, 2.0]])
            ),
            "GENE_MATRIX_GENOTYPE_OUT_OF_RANGE",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.ones((2, 2)), variant_weights=[1.0]
            ),
            "GENE_MATRIX_WEIGHT_COUNT_MISMATCH",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.ones((2, 2)), variant_weights=[1.0, -1.0]
            ),
            "GENE_MATRIX_WEIGHT_NEGATIVE",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.full((2, 1), 2.0), variant_weights=[1e308]
            ),
            "MATRIX_VALUE_NOT_FINITE",
        ),
        (
            lambda: asterism.gene_burden_matrix(
                np.full((2, 1), 2.0), variant_weights=[1e308]
            ),
            "MATRIX_VALUE_NOT_FINITE",
        ),
        (
            lambda: asterism.gene_linear_matrix(np.empty((0, 1))),
            "MATRIX_NO_SUBJECTS",
        ),
    ],
)
def test_builders_refuse_ambiguous_or_malformed_inputs(call, code):
    with pytest.raises(ValueError, match=code):
        call()


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0 + 1.0j]])
            ),
            "GENE_MATRIX_GENOTYPE_NOT_REAL",
        ),
        (
            lambda: asterism.gene_burden_matrix(
                np.array([[1.0]]), variant_weights=[1.0 + 1.0j]
            ),
            "GENE_MATRIX_WEIGHT_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0]]),
                variant_weights=np.array([2**53 + 1], dtype=np.uint64),
            ),
            "GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0, 1.0]]),
                variant_weights=[2**53 + 1, 1.0],
            ),
            "GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64",
        ),
        (
            lambda: asterism.gene_linear_matrix(np.array([0.0, 1.0])),
            "GENE_MATRIX_GENOTYPE_WRONG_SHAPE",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[0.0, 1.0]]), variant_weights=[[1.0, 1.0]]
            ),
            "GENE_MATRIX_WEIGHT_WRONG_SHAPE",
        ),
    ],
)
def test_builders_refuse_lossy_or_wrong_rank_python_inputs(call, code):
    with pytest.raises(ValueError, match=code):
        call()


def test_exact_large_integer_weights_are_not_rejected_merely_for_being_large():
    weight = np.array([2**54], dtype=np.uint64)
    built = asterism.gene_linear_matrix(
        np.array([[1.0], [2.0]]), variant_weights=weight
    )

    scale = float(2**108)
    np.testing.assert_array_equal(
        built, scale * np.array([[1.0, 2.0], [2.0, 4.0]])
    )


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            lambda: asterism.gene_linear_matrix(
                np.ma.array([[0.0, 1.0]], mask=[[False, True]])
            ),
            "GENE_MATRIX_GENOTYPE_MASKED",
        ),
        (
            lambda: asterism.gene_burden_matrix(
                np.array([[0.0, 1.0]]),
                variant_weights=np.ma.array([1.0, 2.0], mask=[False, True]),
            ),
            "GENE_MATRIX_WEIGHT_MASKED",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                [np.ma.array([0.0, 1.0], mask=[False, True])]
            ),
            "GENE_MATRIX_GENOTYPE_MASKED",
        ),
    ],
)
def test_builders_refuse_masked_values_before_numpy_can_expose_them(call, code):
    with pytest.raises(ValueError, match=code):
        call()


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            lambda: asterism.gene_linear_matrix(
                _ArrayHookList(
                    [[0.0]], np.ma.array([[1.0]], mask=[[True]])
                ),
            ),
            "GENE_MATRIX_GENOTYPE_MASKED",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                _ArrayHookList([[0.0]], np.array([[1.0 + 2.0j]]))
            ),
            "GENE_MATRIX_GENOTYPE_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0]]),
                variant_weights=_ArrayHookList(
                    [1.0], np.array([2**53 + 1], dtype=np.uint64)
                ),
            ),
            "GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64",
        ),
    ],
)
def test_builders_validate_the_actual_array_hook_result(call, code):
    with pytest.raises(ValueError, match=code):
        call()


def test_an_ndarray_private_data_hook_cannot_replace_genotypes():
    genotypes = _DataHookArray(
        [[1.0], [2.0]], np.array([[2.0 + 3.0j], [0.0 + 4.0j]])
    )

    built = asterism.gene_linear_matrix(genotypes)

    np.testing.assert_array_equal(built, np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_an_ndarray_private_data_hook_cannot_replace_variant_weights():
    weights = _DataHookArray([1.0], np.array([2**53 + 1], dtype=np.uint64))

    built = asterism.gene_linear_matrix(
        np.array([[1.0], [2.0]]), variant_weights=weights
    )

    np.testing.assert_array_equal(built, np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_the_actual_ndarray_buffer_is_rechecked_when_a_subclass_hides_complex_data():
    genotypes = _DtypeHookArray([[1.0 + 2.0j]], np.float64)

    with pytest.raises(ValueError, match="GENE_MATRIX_GENOTYPE_NOT_REAL"):
        asterism.gene_linear_matrix(genotypes)


def test_the_actual_ndarray_buffer_is_rechecked_when_a_subclass_hides_narrowing():
    weights = _DtypeHookArray(
        np.array([2**53 + 1], dtype=np.uint64), np.float64
    )

    with pytest.raises(ValueError, match="GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64"):
        asterism.gene_linear_matrix(
            np.array([[1.0]]), variant_weights=weights
        )


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[_StatefulInt(1, 2)]], dtype=object)
            ),
            "GENE_MATRIX_GENOTYPE_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0]]),
                variant_weights=np.array([_StatefulInt(1, 2)], dtype=object),
            ),
            "GENE_MATRIX_WEIGHT_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[_StatefulNumpyInt(1, 2)]], dtype=object)
            ),
            "GENE_MATRIX_GENOTYPE_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                _object_array((1, 1), _FloatHookZeroDimensionalArray(1.0, 2.0)),
            ),
            "GENE_MATRIX_GENOTYPE_NOT_REAL",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                np.array([[1.0], [2.0]]),
                variant_weights=_object_array(
                    (1,), _FloatHookZeroDimensionalArray(1.0, 2.0)
                ),
            ),
            "GENE_MATRIX_WEIGHT_NOT_REAL",
        ),
    ],
)
def test_stateful_numeric_objects_cannot_change_after_preflight(call, code):
    with pytest.raises(ValueError, match=code):
        call()
