# Licensed under a 3-clause BSD style license - see LICENSE

from unittest.mock import MagicMock, patch

from feupy.utils.run_fit_and_plot import run_fit_and_plot


def make_inputs():
    datasets = MagicMock()
    datasets.copy.return_value = MagicMock()
    datasets.copy.return_value.names = ["dataset-1"]

    model = MagicMock()
    model.spectral_model.tag = ("SpectralModel", "PL")
    model.copy.return_value = MagicMock()

    fitter = MagicMock()
    result_fit = MagicMock()
    fitter.run.return_value = result_fit

    return datasets, model, fitter, result_fit


def test_run_fit_and_plot_without_plot_or_output():
    datasets, model, fitter, result_fit = make_inputs()

    result = run_fit_and_plot(
        datasets=datasets,
        model=model,
        fitter=fitter,
        show_plot=False,
        show_result_fit=False,
    )

    fitter.run.assert_called_once_with(datasets=datasets)
    assert result is result_fit


@patch("feupy.utils.run_fit_and_plot.calculate_aic", return_value=12.3)
def test_run_fit_and_plot_show_result_fit(mock_calculate_aic, capsys):
    datasets, model, fitter, result_fit = make_inputs()
    result_fit.__str__.return_value = "fit-result"

    run_fit_and_plot(
        datasets=datasets,
        model=model,
        fitter=fitter,
        show_plot=False,
        show_result_fit=True,
    )

    captured = capsys.readouterr()

    assert "fit-result" in captured.out
    assert "AIC: 12.3" in captured.out
    mock_calculate_aic.assert_called_once_with(datasets, result_fit)


@patch("feupy.utils.run_fit_and_plot.plt.show")
@patch("feupy.utils.run_fit_and_plot.SEDPlotter")
@patch(
    "feupy.utils.run_fit_and_plot.build_fp_kwargs",
    return_value={"dataset-1": {"marker": "o"}},
)
def test_run_fit_and_plot_show_plot(
    mock_build_fp_kwargs,
    mock_sed_plotter,
    mock_show,
):
    datasets, model, fitter, _ = make_inputs()
    plotter = mock_sed_plotter.return_value

    run_fit_and_plot(
        datasets=datasets,
        model=model,
        fitter=fitter,
        show_plot=True,
        show_result_fit=False,
        linewidth=2,
    )

    mock_build_fp_kwargs.assert_called_once_with(
        datasets.copy.return_value.names,
        marker_size=4,
    )
    model.copy.assert_called_once_with(name="FIT PL")
    plotter.plot.assert_called_once_with(
        ref_markers={"dataset-1": {"marker": "o"}},
        linewidth=2,
    )
    mock_show.assert_called_once()
