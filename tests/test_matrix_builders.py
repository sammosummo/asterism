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


def test_founder_lineages_make_local_additive_ibd_and_keep_autozygosity():
    built = asterism.local_ibd_matrix(
        np.array([[1, 2], [1, 3], [4, 4]], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        built,
        np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]]),
    )


def test_posterior_lineage_draws_make_the_explicit_weighted_mean():
    draws = np.array(
        [
            [[1, 2], [1, 3], [4, 4]],
            [[1, 2], [3, 4], [4, 4]],
        ],
        dtype=np.int64,
    )
    built = asterism.posterior_local_ibd_matrix(
        draws,
        draw_weights=np.array([1.0, 3.0]),
    )

    np.testing.assert_array_equal(
        built,
        np.array([[1.0, 0.125, 0.0], [0.125, 1.0, 0.75], [0.0, 0.75, 2.0]]),
    )


def test_gene_and_local_constructions_are_positive_semidefinite():
    matrices = [
        asterism.gene_linear_matrix(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
        ),
        asterism.gene_burden_matrix(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
        ),
        asterism.local_ibd_matrix(
            np.array([[1, 2], [1, 3], [4, 4]], dtype=np.int64)
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


def test_local_ibd_is_invariant_to_haplotype_order_and_global_relabelling():
    original = asterism.local_ibd_matrix(
        np.array([[1, 2], [1, 3], [4, 4]], dtype=np.int64)
    )
    relabelled = asterism.local_ibd_matrix(
        np.array([[20, 10], [30, 10], [40, 40]], dtype=np.int64)
    )

    np.testing.assert_array_equal(original, relabelled)


def test_one_posterior_draw_is_the_same_as_its_hard_lineage_matrix():
    lineages = np.array([[1, 2], [1, 3], [4, 4]], dtype=np.int64)
    hard = asterism.local_ibd_matrix(lineages)
    posterior = asterism.posterior_local_ibd_matrix(lineages[None, :, :])

    np.testing.assert_array_equal(hard, posterior)


def test_the_full_uint64_domain_is_available_for_opaque_lineage_labels():
    largest = np.iinfo(np.uint64).max
    built = asterism.local_ibd_matrix(
        np.array([[largest, 1], [largest, 2]], dtype=np.uint64)
    )

    np.testing.assert_array_equal(built, np.array([[1.0, 0.5], [0.5, 1.0]]))


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
        (
            lambda: asterism.local_ibd_matrix(
                np.array([[1, -1], [2, 3]], dtype=np.int64)
            ),
            "LOCAL_IBD_LINEAGE_NOT_POSITIVE",
        ),
        (
            # Nought is the missing code in every pedigree format, and two
            # people carrying it would otherwise read as fully identical by
            # descent and individually autozygous.
            lambda: asterism.local_ibd_matrix(
                np.array([[1, 2], [0, 0]], dtype=np.int64)
            ),
            "LOCAL_IBD_LINEAGE_NOT_POSITIVE",
        ),
        (
            lambda: asterism.local_ibd_matrix(
                np.array([[1, 2], [3, 0]], dtype=np.uint64)
            ),
            "LOCAL_IBD_LINEAGE_NOT_POSITIVE",
        ),
        (
            # Checked in every draw, including one the weights would discard.
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2], [3, 4]], [[1, 2], [0, 0]]], dtype=np.int64),
                draw_weights=[1.0, 0.0],
            ),
            "LOCAL_IBD_LINEAGE_NOT_POSITIVE",
        ),
        (
            lambda: asterism.local_ibd_matrix(
                np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
            ),
            "LOCAL_IBD_LINEAGE_WRONG_SHAPE",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.empty((0, 2, 2), dtype=np.int64)
            ),
            "LOCAL_IBD_NO_DRAWS",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2], [2, 3]]], dtype=np.int64),
                draw_weights=[-1.0],
            ),
            "LOCAL_IBD_WEIGHT_NEGATIVE",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array(
                    [[[1, 2], [2, 3]], [[1, 2], [2, 3]]], dtype=np.int64
                ),
                draw_weights=[1e308, 1e308],
            ),
            "LOCAL_IBD_WEIGHT_SUM_NOT_FINITE",
        ),
    ],
)
def test_builders_refuse_ambiguous_or_malformed_inputs(call, code):
    with pytest.raises(ValueError, match=code):
        call()


def test_noninteger_lineage_labels_are_not_silently_truncated():
    with pytest.raises(ValueError, match="LOCAL_IBD_LINEAGE_NOT_INTEGER"):
        asterism.local_ibd_matrix(np.array([[1.2, 2.0]]))


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
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]]]), draw_weights=[1.0 + 1.0j]
            ),
            "LOCAL_IBD_WEIGHT_NOT_REAL",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]]]),
                draw_weights=np.array([2**53 + 1], dtype=np.uint64),
            ),
            "LOCAL_IBD_WEIGHT_NOT_EXACT_FLOAT64",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]], [[1, 2]]]),
                draw_weights=[2**53 + 1, 1.0],
            ),
            "LOCAL_IBD_WEIGHT_NOT_EXACT_FLOAT64",
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
        (
            lambda: asterism.local_ibd_matrix(np.array([1, 2])),
            "LOCAL_IBD_LINEAGE_WRONG_SHAPE",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[1, 2]])
            ),
            "LOCAL_IBD_LINEAGE_WRONG_SHAPE",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]]]), draw_weights=[[1.0]]
            ),
            "LOCAL_IBD_WEIGHT_WRONG_SHAPE",
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
            lambda: asterism.local_ibd_matrix(
                np.ma.array([[1, 2]], mask=[[False, True]])
            ),
            "LOCAL_IBD_LINEAGE_MASKED",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.ma.array([[[1, 2]]], mask=[[[False, True]]])
            ),
            "LOCAL_IBD_LINEAGE_MASKED",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]]]),
                draw_weights=np.ma.array([1.0], mask=[True]),
            ),
            "LOCAL_IBD_WEIGHT_MASKED",
        ),
        (
            lambda: asterism.gene_linear_matrix(
                [np.ma.array([0.0, 1.0], mask=[False, True])]
            ),
            "GENE_MATRIX_GENOTYPE_MASKED",
        ),
        (
            lambda: asterism.local_ibd_matrix(
                [np.ma.array([1, 2], mask=[False, True])]
            ),
            "LOCAL_IBD_LINEAGE_MASKED",
        ),
        (
            lambda: asterism.posterior_local_ibd_matrix(
                [[np.ma.array([1, 2], mask=[False, True])]]
            ),
            "LOCAL_IBD_LINEAGE_MASKED",
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
        (
            lambda: asterism.local_ibd_matrix(
                _ArrayHookList(
                    [[1, 2]], np.ma.array([[3, 4]], mask=[[False, True]])
                ),
            ),
            "LOCAL_IBD_LINEAGE_MASKED",
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


def test_an_ndarray_private_data_hook_cannot_replace_lineages():
    lineages = _DataHookArray(
        [[1, 2], [1, 3]],
        np.ma.array([[4, 4], [4, 4]], mask=[[False, True], [False, False]]),
    )

    built = asterism.local_ibd_matrix(lineages)

    np.testing.assert_array_equal(built, np.array([[1.0, 0.5], [0.5, 1.0]]))


def test_an_ndarray_private_data_hook_cannot_replace_draw_weights():
    draws = np.array([[[1, 2], [1, 3]], [[1, 2], [3, 4]]])
    weights = _DataHookArray(
        [1.0, 1.0], np.ma.array([1.0, 3.0], mask=[False, True])
    )

    built = asterism.posterior_local_ibd_matrix(
        draws, draw_weights=weights
    )

    np.testing.assert_array_equal(built, np.array([[1.0, 0.25], [0.25, 1.0]]))


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
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2]], [[1, 2]]]),
                draw_weights=np.array(
                    [_StatefulInt(1, 1), _StatefulInt(1, 3)], dtype=object
                ),
            ),
            "LOCAL_IBD_WEIGHT_NOT_REAL",
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
        (
            lambda: asterism.posterior_local_ibd_matrix(
                np.array([[[1, 2], [1, 3]], [[1, 2], [3, 4]]]),
                draw_weights=_object_array(
                    (2,),
                    _FloatHookZeroDimensionalArray(1.0, 1.0),
                    _FloatHookZeroDimensionalArray(1.0, 3.0),
                ),
            ),
            "LOCAL_IBD_WEIGHT_NOT_REAL",
        ),
    ],
)
def test_stateful_numeric_objects_cannot_change_after_preflight(call, code):
    with pytest.raises(ValueError, match=code):
        call()
